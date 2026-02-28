# Criando Um Time de Agentes de IA Generativa Para Teste de Vulnerabilidades em Web Sites

# Importa o módulo para manipulação de variáveis de ambiente e diretórios
import os

# Importa a função para carregar variáveis de ambiente do arquivo .env
from dotenv import load_dotenv

# Importa tipos do módulo 'typing' para melhor definição de estrutura de dados
from typing import TypedDict, Optional

# Importa o modelo de linguagem ChatOpenAI da biblioteca LangChain para uso de LLMs
from langchain_openai import ChatOpenAI

# Importa a ferramenta TavilySearch para realizar pesquisas automáticas na web
from langchain_tavily import TavilySearch

# Importa o componente StateGraph do LangGraph para criação de fluxos entre agentes
from langgraph.graph import StateGraph, END

# Importa funções e classes para criação e execução de agentes com ferramentas
from langchain.agents import AgentExecutor, create_openai_tools_agent

# Importa estrutura para criação de prompts com mensagens contextuais
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Importa tipos de mensagens utilizadas na comunicação entre agentes e humanos
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage 

# Exibe mensagem inicial
print('\nIniciando o Trabalho do Time de Agentes de IA (LangGraph) Para Teste de Vulnerabilidades!\n')

# Carrega as variáveis de ambiente do arquivo .env, sobrescrevendo as existentes se necessário
load_dotenv(override = True)

# Obtém a chave de API da OpenAI do ambiente
openai_api_key = os.getenv("OPENAI_API_KEY")

# Obtém a chave de API da Tavily do ambiente
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Verifica se ambas as chaves de API estão presentes, caso contrário lança erro
if not openai_api_key or not tavily_api_key:
    raise ValueError("Chaves de API da OpenAI ou Tavily não encontradas nas variáveis de ambiente")

# Exibe mensagem confirmando o carregamento das APIs com sucesso
print('APIs Para os LLMs e Ferramentas Carregadas com Sucesso!\n')

# Cria o objeto da ferramenta de busca Tavily, limitando o número máximo de resultados a 3
search_tool = TavilySearch(max_results = 3)

# Inicializa o modelo de linguagem 
llm = ChatOpenAI(api_key = openai_api_key, model = "gpt-4o")

# Define a estrutura do estado compartilhado entre agentes com campos específicos
class AgentState(TypedDict):
    topic: str
    pesquisa: Optional[str]
    analise: Optional[str]
    relatorio_final: Optional[str]
    messages: list[BaseMessage]

# Define função para criação de agentes baseados em LLM e ferramentas específicas
def cria_agente(llm: ChatOpenAI, tools: list, system_prompt: str):
    
    # Cria o template de prompt com contexto do sistema e placeholders para mensagens
    # Para usar o agent OpenAI esse e o contrato (regra) esperada, usar outro agent o padrao seria outro assim como e possivel criar meu proprio loop de prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name = "messages"),
        MessagesPlaceholder(variable_name = "agent_scratchpad"),
    ])
    
    # Cria o agente com base no modelo e nas ferramentas disponíveis
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Define o executor responsável por rodar o agente e registrar logs de execução
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Retorna o executor configurado
    return executor

# Cria o agente pesquisador responsável por buscar informações sobre vulnerabilidades
agente_pesquisador = cria_agente(
    llm,
    [search_tool],
    """Você é um analista sênior em cibersegurança com experiência em detecção de vulnerabilidades e análise de ameaças.
    Sua especialidade é identificar vulnerabilidades comuns e emergentes em websites e sistemas web.
    Sua tarefa é pesquisar o tópico fornecido e reunir informações de fontes confiáveis.
    Para isso, formule consultas de busca concisas e eficazes baseadas no tópico para usar com a ferramenta de pesquisa.
    Ao final, forneça um resumo claro e informativo das descobertas."""
    )

# Cria o agente analista responsável por avaliar e classificar o risco das vulnerabilidades
# Este agente não precisa de ferramenta externa
agente_analista = cria_agente(
    llm,
    [],
    """Você é um especialista em segurança da informação, com foco em avaliação de risco e teste de penetração.
    Sua tarefa é analisar e categorizar as vulnerabilidades encontradas no texto fornecido.
    Avalie o nível de risco (Crítico, Alto, Médio, Baixo) para cada vulnerabilidade e prepare os dados para inclusão no relatório."""
    )

# Cria o agente redator responsável por elaborar o relatório final de segurança
# Este agente não precisa de ferramenta externa
agente_redator = cria_agente(
    llm,
    [], 
    """Você é um redator técnico especializado em relatórios de segurança da informação.
    Sua tarefa é desenvolver um relatório de segurança detalhado com base na análise fornecida.
    O relatório deve ser bem estruturado, claro e incluir as vulnerabilidades, seus níveis de risco e recomendações práticas para mitigação."""
    )

# Define o nó responsável por executar o agente pesquisador
def node_executa_pesquisador(state: AgentState):

    # Exibe mensagem indicando o início da execução do nó do pesquisador
    print("--- NÓ: PESQUISADOR ---")
    
    # Cria uma mensagem humana solicitando pesquisa sobre o tópico fornecido
    messages = [HumanMessage(content = f"Pesquise vulnerabilidades relacionadas a: {state['topic']}")]
    
    # Invoca o agente pesquisador passando as mensagens como entrada
    result = agente_pesquisador.invoke({"messages": messages})

    # Retorna o resultado da pesquisa e adiciona a resposta do agente à lista de mensagens
    return {"pesquisa": result["output"], "messages": messages + [AIMessage(content = result["output"])]}

# Define o nó responsável por executar o agente analista de segurança
def node_executa_analista(state: AgentState):

    # Exibe mensagem indicando o início da execução do nó do analista
    print("--- NÓ: ANALISTA DE SEGURANÇA ---")
    
    # Monta o prompt que será enviado ao agente analista com as descobertas de pesquisa
    prompt = f"""Analise as seguintes vulnerabilidades encontradas e categorize-as por nível de risco.

    Descobertas da Pesquisa:
    {state['pesquisa']}
    """

    # Adiciona a nova mensagem ao histórico existente de mensagens
    current_messages = state["messages"] + [HumanMessage(content = prompt)]
    
    # Invoca o agente analista passando as mensagens atualizadas
    result = agente_analista.invoke({"messages": current_messages})

    # Retorna o resultado da análise e atualiza o histórico de mensagens
    return {"analise": result["output"], "messages": current_messages + [AIMessage(content = result["output"])]}

# Define o nó responsável por executar o agente redator do relatório final
def node_executa_redator(state: AgentState):

    # Exibe mensagem indicando o início da execução do nó do redator
    print("--- NÓ: GERADOR DE RELATÓRIO ---")
    
    # Monta o prompt com a análise anterior para gerar o relatório final de segurança
    prompt = f"""Crie um relatório de segurança detalhado com base na análise abaixo. Inclua vulnerabilidades, riscos e recomendações.

    Análise de Segurança:
    {state['analise']}
    """

    # Atualiza o histórico de mensagens com o novo prompt
    current_messages = state["messages"] + [HumanMessage(content = prompt)]
    
    # Invoca o agente redator com o contexto completo
    result = agente_redator.invoke({"messages": current_messages})

    # Retorna o relatório final e o histórico de mensagens atualizado
    return {"relatorio_final": result["output"], "messages": current_messages + [AIMessage(content = result["output"])]}

# Cria o grafo de estados (workflow) com base na estrutura AgentState
workflow = StateGraph(AgentState)

# Adiciona o nó do pesquisador ao fluxo
workflow.add_node("pesquisador", node_executa_pesquisador)

# Adiciona o nó do analista de segurança ao fluxo
workflow.add_node("analista_seguranca", node_executa_analista)

# Adiciona o nó do redator responsável pelo relatório ao fluxo
workflow.add_node("relatorio_seguranca", node_executa_redator)

# Define o ponto de entrada do fluxo (primeiro nó a ser executado)
workflow.set_entry_point("pesquisador")

# Define as transições entre os nós (ordem de execução)
workflow.add_edge("pesquisador", "analista_seguranca")
workflow.add_edge("analista_seguranca", "relatorio_seguranca")

# Define o término do fluxo após o relatório final
workflow.add_edge("relatorio_seguranca", END)

# Compila o fluxo completo para execução
app = workflow.compile()

# Bloco principal do programa
if __name__ == "__main__":
    
    # Define o tópico a ser analisado pelo time de agentes
    topico = "Segurança em formulários de login no site OWASP Juice Shop (https://owasp.org/www-project-juice-shop/)"
    
    # Exibe mensagem informando que o processo será iniciado
    print('\nTópico Definido. O Time de Agentes (LangGraph) Entrará em Ação!\n')
    
    # Cria o dicionário de entrada inicial para o workflow
    inputs = {"topic": topico, "messages": []}
    
    # Inicializa a variável que armazenará o estado final do fluxo
    final_state = None
    
    # Executa o fluxo e captura o estado em cada etapa
    for output in app.stream(inputs, stream_mode = "values"):
        final_state = output
    
    # Extrai o relatório final do estado gerado
    resultado_final = final_state["relatorio_final"]
    
    # Exibe o relatório final no console
    print("\n\n--- RELATÓRIO FINAL GERADO ---")
    print(resultado_final)

    # Define o nome do arquivo de saída do relatório
    nome_arquivo = "relatorio_seguranca.txt"
    
    try:
        # Abre o arquivo para escrita e salva o relatório com cabeçalho formatado
        with open(nome_arquivo, 'w', encoding = 'utf-8') as f:
            f.write("RELATÓRIO DE ANÁLISE DE VULNERABILIDADES\n")
            f.write("="*40 + "\n\n")
            f.write(f"TÓPICO: {topico}\n\n")
            f.write(resultado_final)
        
        # Exibe mensagem confirmando o sucesso da operação
        print(f"\n✅ Relatório salvo com sucesso no arquivo: {nome_arquivo}")

    # Captura e exibe erros caso ocorram ao salvar o arquivo
    except Exception as e:
        print(f"\n❌ Ocorreu um erro ao salvar o arquivo: {e}")

    # Exibe mensagem final de encerramento do programa
    print('\nObrigado Por Usar o Time de Agentes de IA Para Teste de Vulnerabilidades!\n')




    
