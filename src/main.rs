mod ncu;
mod nsys;
mod trace;

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    version,
    about = "draws what an llm inference engine does inside the gpu"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Convert an nsys sqlite export into kernels.arrow
    Nsys {
        /// the .sqlite file written by `nsys export --type sqlite`
        input: PathBuf,
        /// output file
        #[arg(short, long, default_value = "kernels.arrow")]
        out: PathBuf,
    },
    /// Join an ncu csv export to kernels.arrow and write kernel_metrics.arrow
    Ncu {
        /// csv from `ncu --import r.ncu-rep --csv --page raw --print-units base`
        input: PathBuf,
        /// kernels.arrow of the nsys run made with the same command
        #[arg(long)]
        kernels: PathBuf,
        /// the --launch-skip that ncu was run with
        #[arg(long)]
        skip: usize,
        /// model file to compare dram bytes read per step against
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(short, long, default_value = "kernel_metrics.arrow")]
        out: PathBuf,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Nsys { input, out } => {
            let kernels = nsys::read_kernels(&input)?;
            let batch = trace::kernels_batch(&kernels, "nsys")?;
            trace::write_batch(&out, &batch)?;
            print_summary(&kernels);
            println!("wrote {}", out.display());
        }
        Command::Ncu {
            input,
            kernels,
            skip,
            model,
            out,
        } => {
            let table = ncu::read_csv(&input)?;
            let joined = ncu::join(&table, &kernels, skip)?;
            trace::write_batch(&out, &joined.batch)?;
            print_window(&table.metrics, &joined.per_step, model.as_deref())?;
            println!("wrote {}", out.display());
        }
    }
    Ok(())
}

fn print_summary(kernels: &[trace::KernelRow]) {
    let (Some(first), Some(last)) = (kernels.first(), kernels.iter().map(|k| k.end_ns).max())
    else {
        println!("no kernels");
        return;
    };
    let busy: i64 = kernels.iter().map(|k| k.duration_ns()).sum();
    let mut by_name: HashMap<&str, (usize, i64)> = HashMap::new();
    for k in kernels {
        let e = by_name.entry(&k.name).or_default();
        e.0 += 1;
        e.1 += k.duration_ns();
    }
    let mut top: Vec<_> = by_name.iter().collect();
    top.sort_by_key(|(_, (_, t))| -t);

    let span = last - first.start_ns;
    println!(
        "{} kernels, {} distinct, span {:.3} ms, gpu busy {:.3} ms ({:.1}%)",
        kernels.len(),
        by_name.len(),
        span as f64 / 1e6,
        busy as f64 / 1e6,
        100.0 * busy as f64 / span.max(1) as f64
    );

    let mut per_step: HashMap<u32, usize> = HashMap::new();
    kernels
        .iter()
        .for_each(|k| *per_step.entry(k.step).or_default() += 1);
    let mut sizes: HashMap<usize, usize> = HashMap::new();
    per_step
        .values()
        .for_each(|n| *sizes.entry(*n).or_default() += 1);
    if let Some((size, steps)) = sizes.iter().max_by_key(|(_, n)| **n) {
        println!(
            "{} steps; {steps} of them have {size} kernels each",
            per_step.len()
        );
    }
    for (name, (n, t)) in top.iter().take(8) {
        println!(
            "  {:>6.2}%  {n:>7}x  {name}",
            100.0 * *t as f64 / busy as f64
        );
    }
}

fn print_window(
    metrics: &[String],
    per_step: &[(u32, usize, Vec<u64>)],
    model: Option<&std::path::Path>,
) -> Result<()> {
    let model_bytes = model
        .map(|m| std::fs::metadata(m).map(|md| md.len()))
        .transpose()?;
    let read = metrics.iter().position(|m| m == "dram__bytes_op_read.sum");
    for (step, count, sums) in per_step {
        println!("step {step}: {count} kernels");
        for (m, v) in metrics.iter().zip(sums) {
            println!("  {m:<56} {v:>16}");
        }
        if let (Some(r), Some(size)) = (read, model_bytes) {
            let bytes = sums[r] as f64;
            println!(
                "  dram read / model file = {:.0} / {size} = {:.4}",
                bytes,
                bytes / size as f64
            );
        }
    }
    Ok(())
}
