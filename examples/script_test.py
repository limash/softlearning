from examples.instrument import run_example_local

example_module_name = "examples.development"
example_argv = ("--algorithm=SAC", "--universe=gym", "--domain=HalfCheetah", "--task=v3",
                "--exp-name=my-sac-experiment-1", "--checkpoint-frequency=1000", "--mode=local")

run_example_local(example_module_name, example_argv)
