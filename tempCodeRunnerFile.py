chain = LLMChain(
    llm=model,
    prompt=prompt,
    verbose=True,
    memory=memory
)