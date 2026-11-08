#![forbid(unsafe_code)]
use std::{
    error::Error,
    fs::{File, OpenOptions},
    hint::black_box,
    io::{BufReader, BufWriter, Read, Write},
    time::Instant,
};
use zenpredict::{Model, Predictor};

include!(concat!(env!("OUT_DIR"), "/normalization.rs"));
const MODEL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/head.znpr"));
const CHANNELS: usize = 17;
const CELLS: usize = 32 * 32;
const SLOTS: usize = 128;
type Result<T> = std::result::Result<T, Box<dyn Error>>;

struct Query {
    boxes: [[f32; 4]; SLOTS],
    masks: [[u8; SLOTS]; 2],
}
struct Frame {
    features: [f32; CELLS * CHANNELS],
    content: [f32; 4],
    queries: Vec<Query>,
}

fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}
fn read_f32(reader: &mut impl Read) -> Result<f32> {
    let value = f32::from_bits(read_u32(reader)?);
    if !value.is_finite() || value.abs() > 1e6 {
        return Err("finite bounded input required".into());
    }
    Ok(value)
}
fn valid_box(b: &[f32; 4]) -> bool {
    b.iter().all(|x| (0.0..=1.0).contains(x)) && b[0] <= b[2] && b[1] <= b[3]
}
fn load(path: &str) -> Result<Vec<Frame>> {
    let file = File::open(path)?;
    if file.metadata()?.len() > 128 * 1024 * 1024 {
        return Err("fixture exceeds 128 MiB bound".into());
    }
    let mut reader = BufReader::new(file);
    let mut magic = [0; 8];
    reader.read_exact(&mut magic)?;
    if &magic != b"ZSCH001\0" {
        return Err("incorrect fixture format".into());
    }
    let count = read_u32(&mut reader)? as usize;
    if !(1..=512).contains(&count) {
        return Err("bounded frame count required".into());
    }
    let mut frames = Vec::new();
    frames.try_reserve_exact(count)?;
    for _ in 0..count {
        let mut features = [0.0; CELLS * CHANNELS];
        for value in &mut features {
            *value = read_f32(&mut reader)?;
        }
        let mut content = [0.0; 4];
        for value in &mut content {
            *value = read_f32(&mut reader)?;
        }
        if !valid_box(&content) || content[0] == content[2] || content[1] == content[3] {
            return Err("positive visible content required".into());
        }
        let count = read_u32(&mut reader)? as usize;
        if !(1..=64).contains(&count) {
            return Err("bounded query count required".into());
        }
        let mut queries = Vec::new();
        queries.try_reserve_exact(count)?;
        for _ in 0..count {
            let mut boxes = [[0.0; 4]; SLOTS];
            for b in &mut boxes {
                for value in b.iter_mut() {
                    *value = read_f32(&mut reader)?;
                }
                if !valid_box(b) {
                    return Err("bounded normalized candidate slots required".into());
                }
            }
            let mut masks = [[0; SLOTS]; 2];
            for mask in &mut masks {
                reader.read_exact(mask)?;
                if mask.iter().any(|&x| x > 1) {
                    return Err("binary eligibility mask required".into());
                }
            }
            queries.push(Query { boxes, masks });
        }
        frames.push(Frame {
            features,
            content,
            queries,
        });
    }
    if reader.read(&mut [0])? != 0 {
        return Err("unexpected trailing fixture bytes".into());
    }
    Ok(frames)
}

#[derive(Clone, Copy)]
struct GridRect {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
}
impl GridRect {
    fn cells(self) -> usize {
        self.x1.saturating_sub(self.x0) * self.y1.saturating_sub(self.y0)
    }
    fn intersect(self, other: Self) -> Self {
        Self {
            x0: self.x0.max(other.x0),
            y0: self.y0.max(other.y0),
            x1: self.x1.min(other.x1),
            y1: self.y1.min(other.y1),
        }
    }
}
fn bounds(low: f32, high: f32) -> (usize, usize) {
    let mut begin = 32;
    let mut end = 32;
    for index in 0..32 {
        let center = (index as f32 + 0.5) / 32.0;
        if center >= low && center <= high {
            if begin == 32 {
                begin = index;
            }
            end = index + 1;
        }
    }
    (begin, end)
}
fn grid_rect(b: &[f32; 4]) -> GridRect {
    let (x0, x1) = bounds(b[0], b[2]);
    let (y0, y1) = bounds(b[1], b[3]);
    GridRect { x0, y0, x1, y1 }
}
struct Pool {
    sums: Vec<f64>,
    content: [f32; 4],
    content_grid: GridRect,
    total: [f64; CHANNELS],
}
impl Pool {
    fn new() -> Result<Self> {
        let mut sums = Vec::new();
        sums.try_reserve_exact(33 * 33 * CHANNELS)?;
        sums.resize(33 * 33 * CHANNELS, 0.0);
        Ok(Self {
            sums,
            content: [0.0; 4],
            content_grid: grid_rect(&[0.0; 4]),
            total: [0.0; CHANNELS],
        })
    }
    fn sum(&self, r: GridRect, c: usize) -> f64 {
        if r.cells() == 0 {
            return 0.0;
        }
        let at = |x, y| self.sums[(y * 33 + x) * CHANNELS + c];
        at(r.x1, r.y1) - at(r.x1, r.y0) - at(r.x0, r.y1) + at(r.x0, r.y0)
    }
    fn analyze(&mut self, frame: &Frame) {
        self.sums.fill(0.0);
        for y in 0..32 {
            for x in 0..32 {
                for c in 0..CHANNELS {
                    self.sums[((y + 1) * 33 + x + 1) * CHANNELS + c] =
                        frame.features[(y * 32 + x) * CHANNELS + c] as f64
                            + self.sums[(y * 33 + x + 1) * CHANNELS + c]
                            + self.sums[((y + 1) * 33 + x) * CHANNELS + c]
                            - self.sums[(y * 33 + x) * CHANNELS + c];
                }
            }
        }
        self.content = frame.content;
        self.content_grid = grid_rect(&frame.content);
        for c in 0..CHANNELS {
            self.total[c] = self.sum(self.content_grid, c);
        }
    }
    fn descriptor(&self, b: &[f32; 4]) -> [f32; 62] {
        let [left, top, right, bottom] = self.content;
        let cw = (right - left).max(1e-6);
        let ch = (bottom - top).max(1e-6);
        // Ordered f32 geometry stays independent of the optional normalization math.
        let inside = grid_rect(&[
            left + b[0] * cw,
            top + b[1] * ch,
            left + b[2] * cw,
            top + b[3] * ch,
        ]);
        let intersect = inside.intersect(self.content_grid);
        let outside_count = self.content_grid.cells() - intersect.cells();
        let mut values = [0.0; 62];
        for c in 0..CHANNELS {
            let inner = (self.sum(inside, c) / inside.cells().max(1) as f64) as f32;
            let outer =
                ((self.total[c] - self.sum(intersect, c)) / outside_count.max(1) as f64) as f32;
            values[c] = inner;
            values[17 + c] = outer;
            values[34 + c] = inner - outer;
        }
        let width = (b[2] - b[0]).max(1e-6);
        let height = (b[3] - b[1]).max(1e-6);
        values[51..].copy_from_slice(&[
            b[0],
            b[1],
            b[2],
            b[3],
            width,
            height,
            width * height,
            (width * cw / (height * ch)).ln(),
            (b[0] + b[2]) / 2.0,
            (b[1] + b[3]) / 2.0,
            (cw / ch).ln(),
        ]);
        values
    }
}

fn normalize<const ALGEBRAIC: bool>(values: &[f32; 62]) -> [f32; 62] {
    let mut mean = 0.0_f32;
    for &value in values {
        mean = if ALGEBRAIC {
            mean.algebraic_add(value)
        } else {
            mean + value
        };
    }
    mean /= 62.0;
    let mut variance = 0.0_f32;
    for &value in values {
        let difference = value - mean;
        variance = if ALGEBRAIC {
            variance.algebraic_add(difference.algebraic_mul(difference))
        } else {
            variance + difference * difference
        };
    }
    let inverse = (variance / 62.0 + EPSILON).sqrt().recip();
    let mut output = [0.0; 62];
    for i in 0..62 {
        output[i] = if ALGEBRAIC {
            values[i]
                .algebraic_sub(mean)
                .algebraic_mul(inverse)
                .algebraic_mul(GAMMA[i])
                .algebraic_add(BETA[i])
        } else {
            (values[i] - mean) * inverse * GAMMA[i] + BETA[i]
        };
    }
    output
}

fn infer<const ALGEBRAIC: bool>(
    frames: &[Frame],
    pool: &mut Pool,
    predictor: &mut Predictor<'_>,
    mut writer: Option<&mut dyn Write>,
) -> Result<u64> {
    let mut digest = 14695981039346656037_u64;
    let mut scores = [0.0; SLOTS];
    for frame in frames {
        pool.analyze(frame);
        for query in &frame.queries {
            for (score, b) in scores.iter_mut().zip(&query.boxes) {
                let values = normalize::<ALGEBRAIC>(&pool.descriptor(b));
                *score = predictor.predict(&values)?[0];
                if !score.is_finite() {
                    return Err("nonfinite native query score".into());
                }
                if let Some(output) = writer.as_mut() {
                    output.write_all(&score.to_le_bytes())?;
                }
            }
            for mask in &query.masks {
                let mut choice = u32::MAX;
                let mut best = f32::NEG_INFINITY;
                for (i, (&score, &enabled)) in scores.iter().zip(mask).enumerate() {
                    if enabled == 1 && score > best {
                        best = score;
                        choice = i as u32;
                    }
                }
                for byte in choice.to_le_bytes() {
                    digest = (digest ^ u64::from(byte)).wrapping_mul(1099511628211);
                }
            }
        }
    }
    Ok(digest)
}

fn execute<const ALGEBRAIC: bool>(args: &[String]) -> Result<()> {
    let frames = load(&args[2])?;
    let model = Model::from_bytes_with_schema(MODEL, 0x5A534351303031)?;
    if model.n_inputs() != 62 || model.n_outputs() != 1 {
        return Err("incorrect embedded model dimensions".into());
    }
    let mut predictor = Predictor::new(&model);
    let mut pool = Pool::new()?;
    let requests = frames
        .iter()
        .map(|frame| frame.queries.len() * 2)
        .sum::<usize>();
    if args[1] == "score" {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args[3])?;
        let mut writer = BufWriter::new(file);
        let digest = infer::<ALGEBRAIC>(&frames, &mut pool, &mut predictor, Some(&mut writer))?;
        writer.flush()?;
        println!(
            "{{\"choice_fnv64\":\"{digest}\",\"requests\":{requests},\"embedded_model_bytes\":{},\"face_coverage_certified\":false}}",
            MODEL.len()
        );
    } else if args[1] == "bench" {
        let repeats: usize = args[3].parse()?;
        if !(1..=10000).contains(&repeats) {
            return Err("bounded repeats required".into());
        }
        let digest = infer::<ALGEBRAIC>(&frames, &mut pool, &mut predictor, None)?;
        let start = Instant::now();
        for _ in 0..repeats {
            black_box(infer::<ALGEBRAIC>(
                black_box(&frames),
                &mut pool,
                &mut predictor,
                None,
            )?);
        }
        let elapsed = start.elapsed().as_secs_f64() * 1e9 / (repeats * frames.len()) as f64;
        println!("CHECK {{\"digest\":\"{digest}\",\"requests\":{requests}}}");
        println!(
            "RESULT {{\"inference_ns\":{elapsed},\"embedded_model_bytes\":{},\"sources\":{},\"algebraic\":{ALGEBRAIC}}}",
            MODEL.len(),
            frames.len()
        );
    } else {
        return Err("expected score or bench".into());
    }
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 || !["ordered", "algebraic"].contains(&args[4].as_str()) {
        return Err("usage: query-study score INPUT OUTPUT ordered|algebraic; query-study bench INPUT REPEATS ordered|algebraic".into());
    }
    if args[4] == "algebraic" {
        execute::<true>(&args)
    } else {
        execute::<false>(&args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inclusive_centers_and_constant_region_means() -> Result<()> {
        assert_eq!(bounds(0.5 / 32.0, 0.5 / 32.0), (0, 1));
        let mut frame = Frame {
            features: [7.0; CELLS * CHANNELS],
            content: [0.0, 0.0, 1.0, 1.0],
            queries: Vec::new(),
        };
        let mut pool = Pool::new()?;
        pool.analyze(&frame);
        let descriptor = pool.descriptor(&[0.0, 0.0, 0.5, 0.5]);
        assert_eq!(&descriptor[..34], &[7.0; 34]);
        assert_eq!(&descriptor[34..51], &[0.0; 17]);
        // Clamped content width can let the sample box include cells outside
        // the content mask; outside is content minus the intersection.
        let start = 0.5 / 32.0 - 0.5e-6;
        frame.content = [start, start, start + 1e-7, start + 1e-7];
        pool.analyze(&frame);
        let descriptor = pool.descriptor(&[0.0, 0.0, 1.0, 1.0]);
        assert_eq!(&descriptor[..17], &[7.0; 17]);
        assert_eq!(&descriptor[17..34], &[0.0; 17]);
        Ok(())
    }
}
