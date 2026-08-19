import os, time, json, functools, uuid, warnings
from starlette.exceptions import StarletteDeprecationWarning
import gradio as gr
from tavily import TavilyClient
from linkup import LinkupClient
from ddgs import DDGS
from langfuse import Langfuse
from langfuse.openai import OpenAI #format odpowiedzi zgodny ze standardem OpenAI (nie tylko OpenAI)
from lingua import Language, LanguageDetectorBuilder

#zignoruj ostrzeżenie wynikające ze starej wersji Gradio
warnings.filterwarnings("ignore", category=StarletteDeprecationWarning)

#biblioteki
tavily = TavilyClient(os.getenv("TAVILY_API_KEY"))
linkup = LinkupClient(os.getenv("LINKUP_API_KEY")) if os.getenv("LINKUP_API_KEY") else None #zabezpieczenie przed brakiem klucza
langfuse = Langfuse(host=os.environ.get("LANGFUSE_BASE_URL", os.environ.get("LANGFUSE_HOST")))

#domyślne ustawienia
DEFAULT_MODEL_ID = "ollama (qwen2.5:3b-instruct)"
DEFAULT_SEARCH_PROVIDER = "ddgs"

#niższy priorytet źródeł pochodzących z tłumaczenia
TRANSLATION_PENALTY = 0.75

#szablony promptów
LANGUAGE_TRANSLATOR_PATH = "translator_v2.md"
# SYSTEM_PROMPT_PATH = "response_generator_v4_short.md" #skróć odpowiedź (do testów)
SYSTEM_PROMPT_PATH = "response_generator_v4.md"


#ZMIENNE I FUNKCJE ZWIĄZANE Z JĘZYKAMI
#------------------------------

SOURCE_WEBSITE_LANGUAGE = "pl"
DEFAULT_LANGUAGE = "pl"

#pojedyncze krótkie wiadomości, które są często błędnie klasyfikowane
PROBLEMATIC_LANGUAGE_CLASSIFICATIONS = {
	'jasne': 'pl',
	'okej': 'pl',
	'ok': 'pl',
	'test': 'pl',
	'hello': 'en',
	'thx': 'en',
}

#zapamiętany z ostatniej wiadomości język rozmowy, sprawdzany w razie wątpliwości
established_language = ''

#w kodzie źródłowym zmienne language/target_language/detected_language/itp. zawsze używają kodów 639-1
#funkcja resolve_language zamieniająca kod na pełną nazwę jest używana wyłącznie przed wstawieniem zmiennej do promptu

SUPPORTED_LANGUAGES = { #format: <kod języka>: (<obiekt lingua>, <nazwa wstawiana do promptu>, <tłumaczenie nagłówka 'Źródła'>)
	'pl': (Language.POLISH, 'polski', 'Źródła'),
	'en': (Language.ENGLISH, 'angielski', 'Sources'),
	'uk': (Language.UKRAINIAN, 'ukraiński', 'Джерела'),
	'ar': (Language.ARABIC, 'arabski (standardowy)', 'المصادر'),
	'be': (Language.BELARUSIAN, 'białoruski', 'Крыніцы'),
	'bn': (Language.BENGALI, 'bengalski', 'উৎস'),
	'cs': (Language.CZECH, 'czeski', 'Zdroje'),
	'de': (Language.GERMAN, 'niemiecki', 'Quellen'),
	'es': (Language.SPANISH, 'hiszpański', 'Fuentes'),
	'et': (Language.ESTONIAN, 'estoński', 'Allikad'),
	'fr': (Language.FRENCH, 'francuski', 'Sources'),
	'hi': (Language.HINDI, 'hindi', 'स्रोत'),
	'id': (Language.INDONESIAN, 'indonezyjski', 'Sumber'),
	'it': (Language.ITALIAN, 'włoski', 'Fonti'),
	'ja': (Language.JAPANESE, 'japoński', '出典'),
	'lt': (Language.LITHUANIAN, 'litewski', 'Šaltiniai'),
	'pt': (Language.PORTUGUESE, 'portugalski', 'Fontes'),
	'ru': (Language.RUSSIAN, 'rosyjski', 'Источники'),
	'sk': (Language.SLOVAK, 'słowacki', 'Zdroje'),
	'tr': (Language.TURKISH, 'turecki', 'Kaynaklar'),
	'vi': (Language.VIETNAMESE, 'wietnamski', 'Nguồn'),
	'zh': (Language.CHINESE, 'chiński (mandaryński)', '来源'),
}

def resolve_language(language_code):
	entry = SUPPORTED_LANGUAGES.get(language_code, SUPPORTED_LANGUAGES.get(DEFAULT_LANGUAGE))
	return entry[1] if entry else ''

#biblioteka lingua
LINGUA_CONFIDENCE_FLOOR = 0.3
LINGUA_LANGUAGES = [entry[0] for entry in SUPPORTED_LANGUAGES.values()]

lingua_detector = LanguageDetectorBuilder.from_languages(*LINGUA_LANGUAGES).with_preloaded_language_models().build()


#WCZYTAJ SZABLONY PROMPTÓW NA STARCIE
#------------------------------

with open("prompts/" + SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
	SYSTEM_PROMPT = f.read()
with open("prompts/" + LANGUAGE_TRANSLATOR_PATH, "r", encoding="utf-8") as f:
	LANGUAGE_TRANSLATOR_PROMPT = f.read()


#MODELE LOKALNE (OLLAMA): ETYKIETA -> ID MODELU
#------------------------------

OLLAMA_MODELS = {
	"ollama (qwen2.5:3b-instruct)":					"qwen2.5:3b-instruct",
	"ollama (mistral-7b-instruct)":					"mistral",
	"ollama (bielik-4.5b-v3.0-instruct:Q8_0)":		"SpeakLeash/bielik-4.5b-v3.0-instruct:Q8_0",
}

def resolve_model_id(model_id):
	return OLLAMA_MODELS.get(model_id, model_id)


#WYBÓR KLIENTA
#------------------------------

@functools.lru_cache(maxsize=None)
def client(model_id):
	if model_id == "mistral-small-latest":
		return OpenAI(base_url="https://api.mistral.ai/v1/", api_key=os.getenv("MISTRAL_API_KEY"))
	if model_id == "gpt-4.1-mini":
		return OpenAI(base_url="https://api.openai.com/v1/", api_key=os.getenv("OPENAI_API_KEY"))
	if model_id in OLLAMA_MODELS:
		return OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama")

	return OpenAI(base_url=os.getenv("BIELIK_BASE_URL"), api_key=os.getenv("BIELIK_API_KEY"))


#OBSŁUGA BŁĘDÓW API (WYPADKI BEZ STREAMINGU)
#------------------------------

class ApiError(Exception):
	#wyjątek z komunikatem błędu widocznym dla użytkownika
	pass

def api_call(fn, error_label, *args):
	#wywołaj fn(*args); w razie błędu wywołaj ApiError z opisem
	try:
		return fn(*args)
	except Exception as e:
		raise ApiError(f"**{error_label}**\nSzczegóły: <{e}>") from e

def handle_api_errors(gen_func):
	#dekorator - przechwytuje ApiError z generatora i zwraca jako ChatMessage
	@functools.wraps(gen_func)
	def wrapper(*args, **kwargs):
		try:
			yield from gen_func(*args, **kwargs)
		except ApiError as e:
			yield gr.ChatMessage(content=str(e))
	return wrapper


#WYSZUKIWANIE ŹRÓDEŁ
#------------------------------

@functools.lru_cache(maxsize=500)
def tavily_search(query):
	search = tavily.search(
		query=query,
		search_depth="advanced",
		chunks_per_source=3,
		# include_raw_content=True,
		country="poland",
		include_domains=["gov.pl"]
	)
	return [{"tytul": r["title"], "url": r["url"], "tresc": r["content"], "score": r.get("score", 0)} for r in search["results"]]

@functools.lru_cache(maxsize=500)
def ddgs_search(query):
	results = DDGS().text(
		query=query+" site:gov.pl",
		region="pl-pl",
		max_results=5
	)
	#DDGS nie przypisuje oceny relewancji wynikom - zamiast tego wynik jest ustalany po kolejności
	return [{"tytul": r["title"], "url": r["href"], "tresc": r["body"], "score": 1.0 - i * 0.2} for i, r in enumerate(results)]

@functools.lru_cache(maxsize=500)
def linkup_search(query):
	search = linkup.search(
		query=query,
		depth="standard",
		output_type="searchResults",
		include_domains=["gov.pl"],
		include_images=False
	)
	return [{"tytul": r.name, "url": r.url, "tresc": r.content, "score": 1.0 - i * 0.2} for i, r in enumerate(search.results[:5])]

def filter_out_scores(results):
	return [{k: v for k, v in r.items() if k != "score"} for r in results]

def search(message, search_provider, keep_score=False):
	results = tavily_search(message[:400]) if search_provider == "tavily" else linkup_search(message) if search_provider == "linkup" else ddgs_search(message)
	return results if keep_score else filter_out_scores(results)

def mixed_search(original_query, translated_query, search_provider):
	#wyszukaj 5 źródeł dla oryginalnego zapytania i 5 dla tłumaczenia, wybierz 5 z najwyższym wynikiem
	original_results = search(original_query, search_provider, keep_score=True)
	translated_results = search(translated_query, search_provider, keep_score=True)
	translated_results = [{**r, "score": r["score"] * TRANSLATION_PENALTY} for r in translated_results] #niższy priorytet dla źródeł pochodzących z tłumaczenia

	#usunięcie duplikatów (po URL, zachowaj wynik z wyższym wynikiem)
	by_url = {}
	for r in original_results + translated_results:
		url = r["url"]
		if url not in by_url or r["score"] > by_url[url]["score"]:
			by_url[url] = r

	#wybierz 5 źródeł z najwyższym wynikiem, pozbądź się "score"
	top_results = sorted(by_url.values(), key=lambda r: r["score"], reverse=True)[:5]
	return filter_out_scores(top_results)


#KLASYFIKACJA JĘZYKA I TŁUMACZENIE
#------------------------------

def classify_language(message, trace):
	span = trace.span(name="classify_language", input=message)
	values = lingua_detector.compute_language_confidence_values(message)

	#wyświetl wykryte prawdopodobieństwa (>0.1)
	for v in values:
		if v.value < .1:
			break
		print(f"    {v.language.name}: {v.value:.2f}")

	if not values or values[0].value < LINGUA_CONFIDENCE_FLOOR:
		detected_language = ''
	else:
		lingua_code = values[0].language.iso_code_639_1.name.lower()
		detected_language = lingua_code if lingua_code in SUPPORTED_LANGUAGES else ''

	print(f"=== wykryty język: {detected_language or '(brak)'}")
	span.end(output=detected_language)
	return detected_language


def translate_message(message, target_language, model_id, trace):
	span = trace.span(name="translate_message", input=message)
	result = client(model_id).chat.completions.create(
		model=resolve_model_id(model_id),
		messages=[
			{"role": "system", "content": [{"type": "text", "text": LANGUAGE_TRANSLATOR_PROMPT.replace("{language}", resolve_language(target_language))}]},
			{"role": "user", "content": [{"type": "text", "text": message}]}
		],
		temperature=0.0,
		stream=False,
		trace_id=trace.id
	)
	translated = result.choices[0].message.content.strip()
	span.end(output=translated)
	return translated


#GENEROWANIE ODPOWIEDZI
#------------------------------

def inference(history, trace_id, model_id):
	return client(model_id).chat.completions.create(
		model=resolve_model_id(model_id), #ważne w wypadku localhost/BIELIK_BASE_URL
		messages=history,
		temperature=0.1,
		stream=True,
		stream_options={"include_usage": True},
		trace_id=trace_id,
		stop=["Źródła:", "<|start_header_id|>", "<|eot_id|>"]
	)


@handle_api_errors
def respond(message, history, model_id, search_provider, language_setting, test_mode, session_id):
	#WALIDACJA DANYCH
	#------------------------------

	if not message:
		return
	if not model_id:
		model_id = DEFAULT_MODEL_ID
	if not search_provider:
		search_provider = DEFAULT_SEARCH_PROVIDER
	if not language_setting:
		language_setting = "wykryj automatycznie"

	#jeśli otrzymane session_id jest funkcją, nie wartością -> wywołaj ją, aby ustalić wartość
	if callable(session_id):
		session_id = session_id()


	#WIADOMOŚĆ UŻYTKOWNIKA
	#------------------------------

	message = message.strip()
	trace = langfuse.trace(name="chat-response", session_id=session_id, input=message, tags=[model_id, search_provider], metadata={"model_id": model_id, "search_provider": search_provider})


	#WYKRYWANIE JĘZYKA
	#------------------------------

	if language_setting != "wykryj automatycznie": #język wybrany ręcznie -> pomiń klasyfikację
		detected_language = language_setting
		print(f"=== docelowy język odpowiedzi: {detected_language}")

	else:
		global established_language
		if not history:
			established_language = ''

		if message.lower() in PROBLEMATIC_LANGUAGE_CLASSIFICATIONS: #krótkie, niejednoznaczne wiadomości -> użyj ustalonego języka lub domyślnej klasyfikacji
			detected_language = established_language or PROBLEMATIC_LANGUAGE_CLASSIFICATIONS[message.lower()]
		else:
			detected_language = classify_language(message, trace)
			if not detected_language:
				detected_language = established_language or DEFAULT_LANGUAGE

		#zaktualizuj ustalony język rozmowy
		established_language = detected_language

	#jeśli język inny niż domyślny -> przetłumacz kwerendę przed przeszukaniem źródeł (np. EN -> PL)
	if detected_language != SOURCE_WEBSITE_LANGUAGE:
		translation_bubble = gr.ChatMessage(content="", metadata={"title": "Tłumaczenie treści wiadomości…", "id": 0, "status": "pending", "skip_history": True})
		yield translation_bubble

		start_time_t = time.time()
		search_query = api_call(translate_message, "Błąd tłumaczenia", message, SOURCE_WEBSITE_LANGUAGE, model_id, trace)

		translation_bubble.content = f"Przetłumaczono: '{search_query}'."
		translation_bubble.metadata["status"] = "done"
		translation_bubble.metadata["duration"] = time.time() - start_time_t
		yield translation_bubble

	else:
		translation_bubble = None
		search_query = message


	#RAG
	#------------------------------

	#dymek wyszukiwania
	search_bubble = gr.ChatMessage(content="", metadata={"title": "Wyszukiwanie na stronach Gov.pl…", "id": 1, "status": "pending", "skip_history": True})
	response = [translation_bubble, search_bubble] if translation_bubble else search_bubble
	yield response

	#wyszukiwanie
	start_time = time.time()
	span = trace.span(name="search", input=search_query[:400])
	if translation_bubble: #tłumaczenie -> szukaj źródeł na podstawie oryginalnej wiadomości + jej tłumaczenia
		serp = api_call(mixed_search, "Błąd wyszukiwania", message, search_query, search_provider)
	else:
		serp = api_call(search, "Błąd wyszukiwania", search_query, search_provider) #serp = Search Engine Results Page
	span.end(output=serp)

	#aktualizacja dymku po wyszukiwaniu
	search_bubble.content = f"Znaleziono źródła: {len(serp)}."
	search_bubble.metadata["status"] = "done"
	search_bubble.metadata["duration"] = time.time() - start_time
	yield response

	#prompt systemowy
	current_date = time.strftime("%d.%m.%Y")
	rag_content = json.dumps(serp).encode('utf-8').decode('unicode_escape') #jeśli serp jest pusty -> wynik to []

	system_message = SYSTEM_PROMPT.replace("{current_date}", current_date).replace("{rag_content}", rag_content).replace("{language}", resolve_language(detected_language))

	#ustal pełen kontekst
	history = [{"role": h["role"], "content": h["content"]} for h in history if not (h.get("metadata") or {}).get("skip_history")]	#historia rozmowy (bez system prompt, metadata)
	history.insert(0, {"role": "system", "content": [{"type": "text", "text": system_message}]})									#dodaj prompt systemowy na początek
	history.append({"role": "user", "content": [{"type": "text", "text": message}]})												#dodaj wiadomość użytkownika na końcu

	#wyświetl w konsoli wyszukiwarkę, wysyłaną treść
	print(f"=== wyszukiwarka: {search_provider}")
	print(history)


	#ODPOWIEDŹ
	#------------------------------

	if test_mode:
		#tryb testowy -> wyświetl prompt jako odpowiedź
		response = [*([translation_bubble] if translation_bubble else []), search_bubble, gr.ChatMessage(content=system_message)]
		yield response
		trace.update(output=response[-1].content)
		return

	curr_chunk = ""
	response = [*([translation_bubble] if translation_bubble else []), search_bubble, gr.ChatMessage(content="")]

	try:
		completion = inference(history, trace.id, model_id)
		for chunk in completion:
			if type(chunk) is str:
				curr_chunk = chunk
				response[-1].content += curr_chunk
				yield response
			elif chunk.choices and chunk.choices[0].delta.content is not None:
				curr_chunk = chunk.choices[0].delta.content
				response[-1].content += curr_chunk
				yield response

	except Exception as e:
		response[-1].content += f"\n\n*Błąd generowania odpowiedzi*\nSzczegóły: <{e}>"
		yield response
		return

	if curr_chunk and response[-1].content.endswith(curr_chunk * 2):
		response[-1].content = response[-1].content[:-len(curr_chunk)]
		yield response

	#wyświetl w konsoli odpowiedź (bez źródeł)
	print(response[-1])

	#podaj źródła, jeśli istnieją
	if serp:
		sources_header_translated = SUPPORTED_LANGUAGES.get(detected_language, SUPPORTED_LANGUAGES[DEFAULT_LANGUAGE])[2] #przetłumacz "Źródła'
		response[-1].content += f"\n\n{sources_header_translated}:\n" + "\n".join([f"{i}. [{r['tytul']}]({r['url']})" for i, r in enumerate(serp, 1)])
		yield response

	trace.update(output=response[-1].content)
	langfuse.flush() #wymuś przesłanie danych do Langfuse


#INTERFEJS
#------------------------------

with gr.Blocks() as chat:
	session_id = gr.State(lambda: str(uuid.uuid4())) #ID sesji

	textbox = gr.Textbox(show_label=False, label="Pytanie", placeholder="Zadaj pytanie…", scale=7, autofocus=True, submit_btn=True, stop_btn=True)
	chatbot = gr.Chatbot(label="Czatbot", scale=1, height=400, autoscroll=True, buttons=["copy", "copy_all"])
	chatbot.clear(lambda: str(uuid.uuid4()), outputs=[session_id])

	gr.ChatInterface(
		respond,
		chatbot=chatbot,
		textbox=textbox,
		title="Asystent Interesanta Gov.pl",
		description="Czatbot AI pomagający wyszukiwać aktualne informacje na stronach Gov.pl, oparty o polski otwarty model językowy Bielik 11B v3.0 od SpeakLeash. Stworzony przez Jerzego Głowackiego na licencji Apache 2.0. Sprawdź też <a href='https://ai-gov.pl/widzet.html' target='_blank'>widżet czatu na stronę WWW</a> i <a href='https://huggingface.co/spaces/jglowa/ai-gov.pl/tree/main' target='_blank'>kod źródłowy</a>. Wersja beta, używasz na własną odpowiedzialność.",
		examples=[
			# ["Jak uzyskać NIP?"],
			# ["Jak założyć profil zaufany?"],
			# ["Komu przysługuje zasiłek pielęgnacyjny?"],
			# ["Jak zarejestrować JDG?"]

			["How can I obtain a tax identification number?"],
			["Як я можу отримати PESEL?"],
			["Güvenilir profil nasıl oluşturulur?"],
			["Ako si zaregistrovať živnosť ako fyzická osoba?"]
		],
		cache_examples=False, cache_mode="lazy", save_history=True, analytics_enabled=False, show_progress="full",
		additional_inputs=[
			gr.Dropdown([
				"gpt-4.1-mini",
				"mistral-small-latest",
				"bielik-11b-v3.0-instruct",
				"pllum-12b-chat-2512",
				"ollama (qwen2.5:3b-instruct)",
				"ollama (mistral-7b-instruct)",
				"ollama (bielik-4.5b-v3.0-instruct:Q8_0)",
			], value="bielik-11b-v3.0-instruct", label="Model językowy"),
			gr.Dropdown(["tavily", "linkup", "ddgs"], value="tavily", label="Silnik wyszukiwania"),
			gr.Dropdown(
				[("Wykryj automatycznie", "wykryj automatycznie")]
				+ [(resolve_language(code), code) for code in ["pl", "en", "uk", "de", "tr", "vi"]],
				value="wykryj automatycznie", label="Język"
			),
			gr.Checkbox(value=False, label="Tryb testowy", visible=False), #opcja wyświetlająca ostateczny prompt jako odpowiedź ukryta
			session_id
		]
	)


#START APKI
#------------------------------

chat.launch(
	theme=gr.Theme.load("theme.json"), #lokalna kopia motywu harsh8001/skymist
	css="h1{text-align:center}button.primary{font-size:0 !important}button.primary:after{font-size:var(--button-medium-text-size);content:'Nowy czat'}.contain>.column>.form{order:1}"
)