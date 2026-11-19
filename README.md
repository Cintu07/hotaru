# hotaru 蛍

hotaru draws what an llm inference engine does inside the gpu, and it draws only from real profiler traces. the first engine it reads is llama.cpp running qwen3 on an rtx 5060 laptop gpu. every number in a view comes from a trace row, and each view prints its scale in a corner (for example, 1 particle = 64 kb).

so far hotaru converts nsight systems and nsight compute traces into arrow files. the renderer is not written yet.

## build

you need rust. reading traces does not need a gpu.

```sh
cargo build --release
```

## record a trace

recording needs an nvidia gpu, a cuda build of llama.cpp, nsight systems and nsight compute. these steps were run on windows 11 with cuda 13.3, nsys 2026.1.3 and ncu 2026.2.1.

time every kernel with nsight systems. llama.cpp runs decode steps through cuda graphs, and nsys only shows the kernels inside a graph with `--cuda-graph-trace=node`.

```sh
nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --output run   llama-completion -m Qwen3-8B-Q4_K_M.gguf -ngl 99 -c 4096 -n 250 --ignore-eos --temp 0 -s 1 -no-cnv -p "..."
nsys export --type sqlite --output run.sqlite run.nsys-rep
hotaru nsys run.sqlite --out kernels.arrow
```

`hotaru nsys` prints how many kernels each step ran. a step ends where llama.cpp copies the logits back to the host, so step 0 is the warmup, step 1 the prompt, and every step after that one generated token.

nsight compute replays each kernel, which makes it slow and its timings wrong, so it only counts bytes and instructions for a few steps. run the same command with ncu and skip every kernel before the window. for decode steps 50 to 52 of qwen3 8b that is 865 + 865 + 50 × 690 = 36230 kernels, and the window is 3 × 690.

```sh
ncu --graph-profiling node --launch-skip 36230 --launch-count 2070   --metrics dram__bytes_op_read.sum,dram__bytes_op_write.sum,lts__t_bytes.sum   --export window llama-completion ...same arguments, -n 60
ncu --import window.ncu-rep --csv --page raw --print-units base --print-kernel-base function > window.csv
hotaru ncu window.csv --kernels kernels.arrow --skip 36230 --model Qwen3-8B-Q4_K_M.gguf --out kernel_metrics.arrow
```

`hotaru ncu` stops if any kernel in the window has a different name from the nsys kernel at the same position. on windows, ncu has to run as administrator unless gpu performance counters are opened to all users in the nvidia control panel.

## trace format

a trace is a folder with one arrow ipc file per table.

`kernels.arrow` has one row per kernel launch in start order, with times in nanoseconds on the nsys clock. the other columns hold the step, the stream and the launch configuration as nsys recorded it (grid size, block size, registers, shared memory). kernel names are dictionary encoded because a decode step repeats the same few dozen kernels in every layer.

`kernel_metrics.arrow` has one row per kernel that ncu profiled. `seq` and `step` point back to the matching row of `kernels.arrow`, and every other column is an ncu metric total under its ncu name.

## license

mit, see [LICENSE](LICENSE).
