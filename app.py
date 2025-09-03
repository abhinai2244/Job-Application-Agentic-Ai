

import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

import uuid
import asyncio
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from sentence_transformers import SentenceTransformer, util
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from summarize import summarize_text

# === Graph State ===
from typing import TypedDict, List, Dict

class JobResult(TypedDict):
    description: str
    similarity: float
    cover_letter: str | None
    apply_decision: bool | None  # User's decision to apply
    applied: bool  # Whether the application was successfully submitted

class AgentState(TypedDict):
    user_id: str
    no_jobs: int
    username: str
    password: str
    resume_text: str
    filter_url: str
    job_links: list[str | None]
    current_job_index: int
    job_results: Dict[str, JobResult]  

model = SentenceTransformer("all-MiniLM-L6-v2")

resume_embedding_store = {}

def embed_resume(
    state: AgentState
) -> AgentState:
    """Embed the user's resume and store it by user_id."""
    user_id = state["user_id"]
    resume_text = state["resume_text"]
    embedding = model.encode(resume_text, convert_to_tensor=True)
    resume_embedding_store[user_id] = embedding
    return state


async def get_job_links(state: AgentState) -> AgentState:
    """Log into WorkAtAStartup and return a list of job posting URLs."""
    username = state["username"]
    password = state["password"]
    filter_url = state["filter_url"]

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()

        # Block fonts/images for speed
        await context.route("**/*", lambda route, request: (
            route.abort() if request.resource_type in ["image", "font"] else route.continue_()
        ))

        page = await context.new_page()

        try:
            print("🔐 Navigating to login...")
            await page.goto("https://account.ycombinator.com/?continue=https%3A%2F%2Fwww.workatastartup.com%2F", wait_until= "domcontentloaded")

            await page.fill('input[name="username"]', username)
            await page.fill('input[name="password"]', password)
            await page.click('button[type="submit"]')
            print("✅ Login successful. Current URL:", page.url)

        except Exception as e:
            print(f"❌ Login flow failed: {e}")
            await page.screenshot(path="login_error.png", full_page=True)
            await browser.close()
            return state

        await page.wait_for_timeout(10000)
          
        if page.url != filter_url:
            print("➡️ Navigating to filter URL...")
            try:
                await page.goto(filter_url, timeout=60000)
            except Exception as e:
                print(f"❌ Failed to load filter URL: {e}")
                await browser.close()
                return state
        else:
            print("🟢 Already on filter URL.")

        # Ensure job links are loaded
        try:
            await page.wait_for_selector("a:has-text('View Job')", timeout=10000)
        except PlaywrightTimeoutError:
            print("❌ No job links found.")
            await browser.close()
            return state

        # Infinite scroll
        print("📜 Scrolling to load job listings...")
        for _ in range(30):
            prev_height = await page.evaluate("document.body.scrollHeight")
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(1000)
            new_height = await page.evaluate("document.body.scrollHeight")
            if new_height == prev_height:
                break

        anchors = await page.locator("a:has-text('View Job')").all()
        hrefs = [await a.get_attribute("href") for a in anchors if await a.get_attribute("href")]

        await browser.close()

        state["job_links"] = hrefs[:state["no_jobs"]]  # or more
        print(f"✅ Found {len(state['job_links'])} job links:", state["job_links"])
        return state



async def get_job_description(job_url: str) -> str:
    """Fetch the job description from a specific job page using custom classnames."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            await page.goto(job_url, timeout=45000)
            await page.wait_for_timeout(2000)

            # Extract top title (e.g., company name or job title)
            title = await page.locator(".company-title").all_inner_texts()
            title_text = "\n".join(title).strip()

            # Extract repeated section + prose blocks
            sections = await page.locator(".company-section").all_inner_texts()
            contents = await page.locator(".prose").all_inner_texts()

            # Combine section titles and prose content
            job_description_parts = []
            for sec, con in zip(sections, contents):
                job_description_parts.append(f"## {sec.strip()}\n{con.strip()}")

            full_description = f"# {title_text}\n\n" + "\n\n".join(job_description_parts)
            return full_description

        except Exception as e:
            await page.screenshot(path="job_error.png")
            return f"Error scraping job description: {str(e)}"

        finally:
            await browser.close()


def compare_job_with_resume(job_text: str, user_id: str) -> float:
    """Compare each job with resume using Cosine similarity."""
    if user_id not in resume_embedding_store:
        return 0.0
    job_embedding = model.encode(job_text, convert_to_tensor=True)
    similarity = util.cos_sim(resume_embedding_store[user_id], job_embedding).item()
    return similarity

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4.1-nano")


def generate_cover_letter(resume: str, job_desc: str) -> str:
    """Generate cover letter using OpenAI API."""
    prompt = f"""
    Based on the following resume:
    {resume}

    Write a tailored cover letter of around 200 words for this job description:
    {job_desc}
    """
    cover_letter = llm.invoke(prompt).content
    return str(cover_letter)


async def fetch_all_job_descriptions(state: AgentState) -> AgentState:
    descriptions = {}
    for url in state["job_links"]:
        if url is None: 
            continue 
        desc = await get_job_description(job_url=url)
        descriptions[url] = {"description": desc}
    state["job_results"] = descriptions
    return state


def compare_all_jobs(state: AgentState) -> AgentState:
    for url, entry in state["job_results"].items():
        desc = entry["description"]
        similarity = compare_job_with_resume(
            job_text=desc,
            user_id=state["user_id"]
        )
        entry["similarity"] = similarity
    return state


def generate_all_cover_letters(state: AgentState) -> AgentState:
    for url, entry in state["job_results"].items():
        if entry["similarity"] > 0.4:
            print(f"📝 Generating cover letter for {url} (similarity: {entry['similarity']:.2f})")
            
            summarized_resume = summarize_text(state["resume_text"], 10)
            summarized_job_desc = summarize_text(entry["description"], 10)
            
            cover_letter = generate_cover_letter(
                resume=summarized_resume,
                job_desc=summarized_job_desc
            )
            
            entry["cover_letter"] = cover_letter
            entry["apply_decision"] = None  
            entry["applied"] = False        

        else:
            print(f"❌ Skipping {url} (similarity: {entry['similarity']:.2f}) — not a good match")
            entry["cover_letter"] = "Not a good match"
            entry["apply_decision"] = False
            entry["applied"] = False

    return state


async def should_apply_to_job(state: AgentState) -> AgentState:
    for url, entry in state["job_results"].items():
        if entry.get("apply_decision") is None and entry.get("cover_letter") != "Not a good match":
            print(f"\n🔗 Job URL: {url}")
            print(f"\n📄 Cover Letter:\n{entry['cover_letter']}\n")

            while True:
                user_input = input("👉 Do you want to apply to this job? (y/n): ").strip().lower()
                if user_input in ["y", "yes"]:
                    entry["apply_decision"] = True
                    break
                elif user_input in ["n", "no"]:
                    entry["apply_decision"] = False
                    break
                else:
                    print("⚠️ Please enter 'y' or 'n'.")

    return state


async def auto_apply_to_job(state: AgentState) -> AgentState:
    username = state["username"]
    password = state["password"]
    job_results = state["job_results"]

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        # Block fonts/images for speed
        await page.route("**/*", lambda route, request: (
            route.abort() if request.resource_type in ["image", "font"] else route.continue_()
        ))

        try:
            print("🔐 Navigating to login...")
            await page.goto("https://account.ycombinator.com/?continue=https%3A%2F%2Fwww.workatastartup.com%2F", wait_until= "domcontentloaded")

            await page.fill('input[name="username"]', username)
            await page.fill('input[name="password"]', password)
            await page.click('button[type="submit"]')
            print("✅ Login successful. Current URL:", page.url)

        except Exception as e:
            print(f"❌ Login flow failed: {e}")
            await page.screenshot(path="login_error.png", full_page=True)
            await browser.close()
            return state

        await page.wait_for_timeout(10000)

        for url, entry in job_results.items():
            if entry.get("apply_decision") is not True:
                continue

            print(f"🚀 Applying to job: {url}")
            
            if page.url != url:
               try:
                   await page.goto(url, timeout=10000)
                   await page.wait_for_selector("text=Apply", timeout=10000)
                   await page.click("text=Apply")
   
                   # Wait for modal with textarea and Send button
                   await page.wait_for_selector("textarea", timeout=10000)
                   await page.fill("textarea", str(entry["cover_letter"]))
                   await page.click("button:has-text('Send')")
                   await page.wait_for_timeout(5000)  # Wait for application to process
                   entry["applied"] = True
                   print("✅ Successfully applied.")
               except PlaywrightTimeoutError:
                   print(f"❌ Failed to apply to job: {url}")
                   entry["applied"] = False
               except Exception as e:
                   print(f"⚠️ Unexpected error for {url}: {e}")
                   entry["applied"] = False
            else: 
                print(f"🟢 Already on job page: {url}")

        await browser.close()
        return state


# === Graph Definition ===
workflow = StateGraph(AgentState)

workflow.add_node("EmbedResume", RunnableLambda(embed_resume))
workflow.add_node("GetJobLinks", RunnableLambda(get_job_links))
workflow.add_node("FetchJobDesc", RunnableLambda(fetch_all_job_descriptions))
workflow.add_node("Compare", RunnableLambda(compare_all_jobs))
workflow.add_node("GenCover", RunnableLambda(generate_all_cover_letters))
workflow.add_node("ShouldApplyToJob", RunnableLambda(should_apply_to_job))
workflow.add_node("AutoApplyToJob", RunnableLambda(auto_apply_to_job))



workflow.set_entry_point("EmbedResume")
workflow.add_edge("EmbedResume", "GetJobLinks")
workflow.add_edge("GetJobLinks", "FetchJobDesc")
workflow.add_edge("FetchJobDesc", "Compare")
workflow.add_edge("Compare", "GenCover")
workflow.add_edge("GenCover", "ShouldApplyToJob")
workflow.add_edge("ShouldApplyToJob", "AutoApplyToJob")
workflow.add_edge("AutoApplyToJob", END)

graph = workflow.compile()

async def run_graph(pdf_text, user_id, username, password, filter_url, no_jobs):
    result = await graph.ainvoke({
        "user_id": user_id,
        "no_jobs": no_jobs, 
        "username": username,
        "password": password,
        "resume_text": pdf_text,
        "filter_url": filter_url,
        "current_job_index": 0,
        "job_results": {}
    })
    print("✅ Graph execution completed successfully!")
    for idx, (url, job) in enumerate(result['job_results'].items(), 1):
        print(f"\nJob {idx}:")
        print(f"URL: {url}")
        print(f"Similarity Score: {job.get('similarity', 'N/A')}")
        print(f"Applied: {'Yes' if job.get('applied', False) else 'No'}")
        print("Cover Letter:")
        print(job.get('cover_letter', 'No cover letter generated'))
        print("-" * 40)

if __name__ == "__main__":
    import getpass
    from read_pdf import read_pdf

    username = input("Enter your ycombinator username: ")
    password = getpass.getpass("Enter your ycombinator password: ")
    filter_url = input("Enter the Job Portal URL (e.g., https://www.workatastartup.com/companies?demographic=any&hasEquity=any&hasSalary=any&industry=any&interviewProcess=any&jobType=fulltime&layout=list-compact&minExperience=0&minExperience=1&remote=yes&role=eng&role_type=fe&role_type=fs&sortBy=created_desc&tab=any&usVisaNotRequired=any): ")
    resume_file = input("Enter your resume file name (e.g., resume.pdf): ")
    while True:
        try:
            no_jobs = int(input("Enter the number of jobs to fetch (max 10, default 10): ") or 10)
            if no_jobs <= 10:
                break
            else:
                print("Please enter a number less than or equal to 10.")
        except ValueError:
            print("Please enter a valid integer.")

    user_id = str(uuid.uuid4())
    pdf_text = read_pdf(resume_file)

    asyncio.run(run_graph(pdf_text, user_id, username, password, filter_url, no_jobs))

