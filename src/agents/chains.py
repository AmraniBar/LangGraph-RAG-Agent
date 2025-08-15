from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain import hub


def create_rag_chain(llm):
    # More flexible RAG prompt for better generation
    # Simple direct prompt that works better with smaller models
    rag_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                """Based on the following context, answer this question: {question}

Context: {context}

Answer:""",
            )
        ]
    )

    return rag_prompt | llm | StrOutputParser()


# def create_question_rewriter(llm):
#     rewriter_system_prompt = """You are a question re-writer that converts an input question into a better optimized version for vector store retrieval document.
# You are given both a question and a document.
# - First, check if the question is relevant to the document by identifying a connection or relevance between them.
# - If there is a little relevancy, rewrite the question based on the semantic intent of the question and the context of the document.
# - If no relevance is found, simply return this single word "question not relevant." dont return the entire phrase
# Your goal is to ensure the rewritten question aligns well with the document for better retrieval."""

#     rewrite_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", rewriter_system_prompt),
#             (
#                 "human",
#                 """Here is the initial question: \\n\\n {question} \\n,
#              Here is the document: \\n\\n {documents} \\n ,
#              Formulate an improved question. if possible other return 'question not relevant'.""",
#             ),
#         ]
#     )
#     return rewrite_prompt | llm | StrOutputParser()
