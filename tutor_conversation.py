"""Create a ConversationalRetrievalChain for question/answering."""
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ConversationChain

from langchain.chat_models.openai import ChatOpenAI
from langchain.memory.buffer import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
import openai

COURSE_INTRODUCTION_MESSAGE = """Benvenutu in u mio corsu! Sò cuntéintu di accuglie in u mio corsu di lingua corsa. U mio scopu hè di t'aiutà à scopre è imparà a biddezza di a lingua corsa.
Tradution: Benvenutu in u mio corsu! Je suis ravi de t'accueillir dans mon cours de langue corse. Mon rôle est de t'aider à découvrir et à apprendre la magnifique langue corse.
\n
Per principià, mi vulerìa valutà u to livellu di cunniscenza di a lingua corsa. Quale hè a to relazione cù a Corsica? Hai dighjà studiatu a lingua in passatu o hè a prima volta ? Raccuntami un pocu di più nantu à te è à e to motivazioni.
Traduction: Pour commencer, j'aimerais évaluer ton niveau de connaissance de la langue corse. Quelle est ta relation avec la Corse ? Est-ce que tu as déjà étudié la langue auparavant ou est-ce que c'est la première fois ? Raconte-moi un peu plus sur toi et tes motivations.
"""


def get_tutor_chain(
    stream_handler: BaseCallbackHandler,
    conversation_memory: ConversationBufferMemory,
    prompt_template: PromptTemplate,
    tracing: bool = False,
) -> ConversationChain:
    """Create a ConversationalChain to teach a language."""

    manager = AsyncCallbackManager([])

    stream_manager = AsyncCallbackManager([stream_handler])
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    streaming_llm = ChatOpenAI(
        client=openai.ChatCompletion,
        streaming=True,
        model_name="gpt-4",
        callback_manager=stream_manager,
        verbose=True,
        temperature=0,
    )

    chat_chain = ConversationChain(
        memory=conversation_memory,
        llm=streaming_llm,
        callback_manager=manager,
        prompt=prompt_template,
        verbose=True,
    )
    return chat_chain
