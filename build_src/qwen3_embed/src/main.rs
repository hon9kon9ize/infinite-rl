use anyhow::anyhow;
use candle_core::{DType, Device};
use clap::Parser;
use std::path::Path;
use std::time::Instant;

mod qwen3;
use qwen3::Qwen3TextEmbedding;

#[derive(Parser)]
#[command(name = "qwen3_embed")]
#[command(about = "Generate text embeddings using Qwen3-Embedding-0.6B model")]
struct Args {
    /// The document text to embed
    #[arg(short, long)]
    document: Option<String>,

    /// A query text to embed
    #[arg(short, long)]
    query: Option<String>,

    /// Only download and cache model files, do not run embedding
    #[arg(long, default_value_t = false)]
    cache_only: bool,

    /// Optional cache directory (overrides the default `qwen3_local_cache`)
    #[arg(long)]
    cache_dir: Option<String>,
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0f32;
    let mut na = 0f32;
    let mut nb = 0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    real_main().await
}

#[cfg(target_arch = "wasm32")]
#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    real_main().await
}

async fn real_main() -> anyhow::Result<()> {
    let args = Args::parse();
    let t0 = Instant::now();
    // Use --cache-dir if provided, otherwise default to ./qwen3_local_cache
    let cache_dir = if let Some(ref dir) = args.cache_dir {
        Path::new(dir)
    } else {
        Path::new("qwen3_local_cache")
    };

    // 1. Handle Cache-Only Request
    if args.cache_only {
        eprintln!("--cache-only requested: populating cache and exiting");

        if cache_dir.exists() {
            eprintln!("Cache already present at {}", cache_dir.display());
            return Ok(());
        }

        #[cfg(feature = "hf-hub")]
        {
            Qwen3TextEmbedding::from_hf_cached(
                "Qwen/Qwen3-Embedding-0.6B",
                &Device::Cpu,
                DType::F32,
                512,
                cache_dir,
            )
            .map_err(|e| anyhow!(e.to_string()))?;
            eprintln!("Cache populated at {}", cache_dir.display());
            return Ok(());
        }

        #[cfg(not(feature = "hf-hub"))]
        {
            return Err(anyhow!(
                "--cache-only requested but hf-hub feature is disabled; cannot populate cache in this build. Use --cache-dir to point to a cache directory."
            ));
        }
    }

    // 2. Initialize Model (Timed)
    let t_model = Instant::now();

    let model = if cache_dir.exists() {
        Qwen3TextEmbedding::from_local_cache(&Device::Cpu, DType::F32, cache_dir)
            .map_err(|e| anyhow!(e.to_string()))?
    } else {
        #[cfg(feature = "hf-hub")]
        {
            Qwen3TextEmbedding::from_hf_cached(
                "Qwen/Qwen3-Embedding-0.6B",
                &Device::Cpu,
                DType::F32,
                512,
                cache_dir,
            )
            .map_err(|e| anyhow!(e.to_string()))?
        }
        #[cfg(not(feature = "hf-hub"))]
        {
            return Err(anyhow!(
                "No cache found and hf-hub is disabled. Please provide the cache directory (e.g. `qwen3_local_cache`) or run with --cache-dir to point to an existing cache: {}",
                cache_dir.display()
            ));
        }
    };

    // --- OPTIONAL WARMUP ---
    // model.embed(&["warmup"])?; 
    
    let model_ms = t_model.elapsed().as_millis() as u64;

    // 3. Prepare Inputs
    let document = args
        .document
        .as_deref()
        .ok_or_else(|| anyhow!("--document is required"))?;
    let query = args
        .query
        .as_deref()
        .ok_or_else(|| anyhow!("--query is required"))?;

    let inputs = vec![format!("query: {}", query), format!("passage: {}", document)];

    // 4. Embed (Timed)
    let t_embed = Instant::now();
    let embeddings = model.embed(&inputs).map_err(|e| anyhow!(e.to_string()))?;
    let embed_ms = t_embed.elapsed().as_millis() as u64;

    // 5. Compute Similarity (Timed)
    let t_sim = Instant::now();
    let sim = cosine_sim(&embeddings[0], &embeddings[1]);
    let sim_ms = t_sim.elapsed().as_millis() as u64;

    let total_ms = t0.elapsed().as_millis() as u64;

    // 6. Output Results
    let out = serde_json::json!({
        "cosine_similarity": sim,
        "timings_ms": {
            "model_load": model_ms,
            "embed": embed_ms,
            "similarity": sim_ms,
            "total": total_ms
        }
    });

    eprintln!(
        "timings (ms): model={} embed={} sim={} total={}",
        model_ms, embed_ms, sim_ms, total_ms
    );
    println!("{}", serde_json::to_string_pretty(&out)?);

    Ok(())
}