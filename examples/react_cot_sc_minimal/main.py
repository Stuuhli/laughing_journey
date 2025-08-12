from react_cot_sc import ReActCoTSCAgent

def main() -> None:
    agent = ReActCoTSCAgent(model="llama3", embedding_model_name="demo", k=2)
    query = input("Frage: ")
    print(agent.run(query))

if __name__ == "__main__":
    main()
