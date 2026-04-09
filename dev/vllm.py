from vllm import LLM, SamplingParams

# Load your model
llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2")

# Set your sampling parameters, including the seed
# Note: You must have a temperature > 0 for the seed to actually affect sampling
sampling_params = SamplingParams(
    temperature=0.7, 
    top_p=0.9, 
    seed=42  # <--- Here is your seed
)

prompts = ["The best way to cook a steak is"]

# Generate the output
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)