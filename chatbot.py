from transformers import pipeline

def configurar_modelo():
    print("Carregando o modelo GPT-Neo 1.3B...")
    # Carregando o modelo GPT-Neo a partir da biblioteca Transformers
    modelo = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")
    return modelo


# Definição do contexto de tecnologia
contexto_tecnologia = """
Tecnologia é o estudo e aplicação de ferramentas, sistemas e processos que facilitam tarefas e oferecem soluções para problemas. Exemplos de áreas da tecnologia incluem:
1. Inteligência artificial (IA), que desenvolve máquinas inteligentes que simulam comportamentos humanos.
2. Computação em nuvem, que permite acesso a dados e aplicativos remotamente.
3. Robótica, voltada para a criação de máquinas autônomas.
4. Inovações digitais, como smartphones, automação e IoT (Internet das Coisas).
"""

def apresentar_contexto():
    print("Bem-vindo ao Chatbot de Tecnologia!")
    print(f"Contexto inicial:\n{contexto_tecnologia}\n")


def responder_pergunta(modelo, pergunta):
    # Adiciona um prompt claro e objetivo relacionado à tecnologia
    prompt = f"""
    Você é um especialista em tecnologia. Responda com clareza e objetividade.
    Contexto: {contexto_tecnologia}
    Pergunta: {pergunta}
    Resposta:
    """
    # Gera a resposta com base no prompt
    resposta = modelo(
        prompt,
        max_new_tokens=100,  # Número máximo de tokens gerados para a resposta
        num_return_sequences=1,
        truncation=True  # Corta entradas muito longas
    )[0]["generated_text"]
    # Remove o texto inicial do prompt da resposta gerada
    return resposta[len(prompt):].strip()


# Função principal do chatbot
def executar_chatbot():
    modelo = configurar_modelo()  # Configuração do modelo
    apresentar_contexto()  # Apresentação inicial

    perguntas_respondidas = []  # Para salvar as interações do usuário

    for i in range(3):  # Permitir três perguntas
        pergunta = input(f"Pergunta {i+1}: ")
        resposta = responder_pergunta(modelo, pergunta)
        print(f"Resposta: {resposta}\n")
        perguntas_respondidas.append(f"Pergunta {i+1}: {pergunta}\nResposta: {resposta}")

    # Exibir resumo ao final
    print("\nResumo das respostas:")
    for resumo in perguntas_respondidas:
        print(resumo)

if __name__ == "__main__":
    executar_chatbot()

