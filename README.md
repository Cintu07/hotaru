# hotaru 蛍

hotaru draws what an llm inference engine does inside the gpu, and it draws only from real profiler traces. the first engine it reads is llama.cpp running qwen3 on an rtx 5060 laptop gpu. every number in a view comes from a trace row, and each view prints its scale in a corner (for example, 1 particle = 64 kb).

so far hotaru converts nsight systems traces into arrow files. the renderer is not written yet.

## build

you need rust. reading traces does not need a gpu.

```sh
cargo build --release
```

## use

profile the engine with nsight systems, export the report to sqlite, then convert it:

```sh
nsys export --type sqlite --output run.sqlite run.nsys-rep
hotaru nsys run.sqlite --out kernels.arrow
```

it prints the kernel count, how busy the gpu was, and the kernels that took the most time.

## trace format

a trace is a folder with one arrow ipc file per table. each row of `kernels.arrow` is one kernel launch, in start order, with times in nanoseconds on the nsys clock. the other columns hold the stream and the launch configuration as nsys recorded it (grid size, block size, registers, shared memory). kernel names are dictionary encoded because a decode step repeats the same few dozen kernels in every layer.

## license

mit, see [LICENSE](LICENSE).
