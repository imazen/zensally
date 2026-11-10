#![forbid(unsafe_code)]
use std::{env, fs, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo::rerun-if-env-changed=ZENSALLY_QUERY_HEAD");
    let path = PathBuf::from(env::var("ZENSALLY_QUERY_HEAD")?);
    println!("cargo::rerun-if-changed={}", path.display());
    let value: serde_json::Value = serde_json::from_str(&fs::read_to_string(path)?)?;
    if value["format"] != "zensally-query-head-v1" {
        return Err("unsupported query head".into());
    }
    let destination = PathBuf::from(env::var("OUT_DIR")?);
    let baked = zenpredict_bake::bake_from_json_str(&serde_json::to_string(&value["bake"])?)?;
    fs::write(destination.join("head.znpr"), baked)?;
    let mut code = String::new();
    for (key, constant) in [("gamma", "GAMMA"), ("beta", "BETA")] {
        let values = value[key]
            .as_array()
            .ok_or("normalization array required")?;
        if values.len() != 62 {
            return Err("62 normalization values required".into());
        }
        code.push_str(&format!("const {constant}: [f32; 62] = ["));
        for number in values {
            let number = number.as_f64().ok_or("numeric normalization required")? as f32;
            if !number.is_finite() {
                return Err("finite normalization required".into());
            }
            code.push_str(&format!("f32::from_bits({}),", number.to_bits()));
        }
        code.push_str("];\n");
    }
    let epsilon = value["epsilon"].as_f64().ok_or("epsilon required")? as f32;
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err("finite positive epsilon required".into());
    }
    code.push_str(&format!(
        "const EPSILON: f32 = f32::from_bits({});\n",
        epsilon.to_bits()
    ));
    fs::write(destination.join("normalization.rs"), code)?;
    Ok(())
}
