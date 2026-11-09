#![forbid(unsafe_code)]
use std::{env, fs, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo::rerun-if-env-changed=ZENSALLY_RESIDUAL_MODEL");
    let path = env::var("ZENSALLY_RESIDUAL_MODEL")?;
    println!("cargo::rerun-if-changed={path}");
    let value: serde_json::Value = serde_json::from_str(&fs::read_to_string(path)?)?;
    if value["format"] != "zensally-compact-residual-v1" {
        return Err("unsupported model".into());
    }
    for name in ["base", "residual"] {
        let baked = zenpredict_bake::bake_from_json_str(&serde_json::to_string(&value[name])?)?;
        fs::write(
            PathBuf::from(env::var("OUT_DIR")?).join(format!("{name}.znpr")),
            baked,
        )?;
    }
    Ok(())
}
