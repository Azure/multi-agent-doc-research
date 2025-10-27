#!/usr/bin/env python3
"""
Comprehensive Batch Evaluation Script for Microsoft Product Inquiry Chatbot
This script generates and evaluates the chatbot's performance on Microsoft product queries using Azure evaluation SDK.

Usage:
    python batch_eval.py --input data/RTF_queries.csv --max_concurrent 3 --max_tokens 2000 --temperature 0.5 --query_rewrite true --plan_execute true
"""

import asyncio
import csv
import json
import logging
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse
import os
import pytz
from azure.ai.evaluation import evaluate
from azure.ai.evaluation import (
    RelevanceEvaluator,
    SimilarityEvaluator,
    RetrievalEvaluator,
)
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np


def setup_comprehensive_logging(verbose: bool = False):

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_level = logging.DEBUG if verbose else logging.INFO

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    root_logger.info("🚀 initialized logging ")
    root_logger.info(f"📊 log level: {logging.getLevelName(log_level)}")

    return root_logger


logger = setup_comprehensive_logging()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Settings
from services.orchestrator import Orchestrator
from services.plan_executor import PlanExecutor
from services.search_crawler import GoogleSearchCrawler, BingSearchCrawler
from services.bing_grounding_search import BingGroundingSearch, BingGroundingCrawler
from model.models import ChatMessage
from utils.enum import SearchEngine


class BatchEvaluator:
    """Microsoft query evaluator for Microsoft product inquiry chatbot"""

    def __init__(self, settings: Settings):

        logger.info("Setup BatchEvaluator...")
        self.settings = settings

        # Azure OpenAI 설정 정보 로깅
        logger.info("Azure OpenAI Configuration:")
        logger.info(f"   • endpoint: {settings.AZURE_OPENAI_ENDPOINT}")
        logger.info(f"   • deployment: {settings.AZURE_OPENAI_DEPLOYMENT_NAME}")
        logger.info(f"   • API version: {settings.AZURE_OPENAI_API_VERSION}")

        model_config = {
            "azure_endpoint": settings.AZURE_OPENAI_ENDPOINT,
            "api_key": settings.AZURE_OPENAI_API_KEY,
            "azure_deployment": settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            "api_version": settings.AZURE_OPENAI_API_VERSION,
            "type": "azure_openai",
        }

        logger.info("🔧 Creating evaluators...")
        self.retrieval_evaluator = RetrievalEvaluator(model_config)
        self.relevance_evaluator = RelevanceEvaluator(model_config)
        self.similarity_evaluator = SimilarityEvaluator(model_config)
        self.eval_input_path = "data/evaluation_input.jsonl"
        self.eval_output_path = "evals/results/evaluation_results.json"

        logger.info("✅ BatchEvaluator successfully initialized")

    def batch_evaluate(
        self,
        eval_input_path: Optional[str] = None,
        eval_output_path: Optional[str] = None,
    ) -> str:
        """
        Evaluate batch responses for the provided queries

        Args:
            eval_input_path: Path to the input file containing queries in JSONL format
            eval_output_path: Path to the output file to save evaluation results

        Returns:
            eval_output_path: Path to the file containing evaluation results in JSON format
        """
        logger.info("🔍 Batch evaluation started...")
        self.eval_input_path = eval_input_path
        self.eval_output_path = eval_output_path

        logger.info(f"📁 Input file: {eval_input_path}")
        logger.info(f"📁 Output file: {eval_output_path}")

        column_mapping = {
            "query": "${data.query}",
            "response": "${data.response}",
        }

        logger.info("⚙️ Configuring evaluation parameters...")
        logger.info("   • Evaluator: relevance")
        logger.info("   • Column mapping configured")

        timezone = pytz.timezone("Asia/Seoul")
        evaluation_name = (
            f"evaluation_{datetime.now(tz=timezone).strftime('%Y-%m-%d_%H-%M-%S')}"
        )
        logger.info(f"📝 Evaluation name: {evaluation_name}")

        try:
            logger.info("🚀 Azure AI Evaluation SDK Running...")
            result = evaluate(
                evaluation_name=evaluation_name,
                data=self.eval_input_path,
                evaluators={
                    "relevance": self.relevance_evaluator,
                },
                evaluator_config={
                    "relevance": {"column_mapping": column_mapping},
                },
                output_path=self.eval_output_path,
            )

            logger.info("✅ Batch evaluation successfully completed")

            return self.eval_output_path

        except Exception as e:
            logger.error(f"Error occurred during batch evaluation: {e}")
            raise


class BatchResponseGenerator:
    """Batch response generator for Microsoft product inquiry chatbot"""

    def __init__(
        self,
        settings: Settings,
        search_crawler: Optional[BingSearchCrawler] = None,
        bing_grounding_search: Optional[BingGroundingSearch] = None,
        search_engine: SearchEngine = SearchEngine.BING_SEARCH_CRAWLING,
        query_rewrite: bool = True,
        plan_execute: bool = True,
        interval_seconds: float = 1.0,
    ):
        """Initialize batch response generator"""
        logger.info("🚀 Initializing BatchResponseGenerator...")
        self.settings = settings

        logger.info("🔧 Configuring Redis settings...")
        redis_config = {
            "host": settings.REDIS_HOST,
            "port": settings.REDIS_PORT,
            "password": settings.REDIS_PASSWORD,
            "db": settings.REDIS_DB,
            "decode_responses": True,
        }

        if search_crawler is not None:
            self.search_crawler = search_crawler
            logger.info("✅ Custom search engine in use")
        else:
            self.search_crawler = BingSearchCrawler(redis_config=redis_config)
            logger.info("✅ Default Bing search engine in use")

        if bing_grounding_search is not None:
            self.bing_grounding_search = bing_grounding_search
            logger.info("✅ Custom Bing grounding search engine in use")
        else:
            self.bing_grounding_search = BingGroundingSearch(redis_config=redis_config)
            logger.info("✅ Default Bing ground search engine in use")

        logger.info("🔧 Initializing orchestrator and plan executor...")
        self.orchestrator = Orchestrator(settings=settings)
        self.plan_executor = PlanExecutor(settings=settings)
        self.query_rewrite = query_rewrite
        self.plan_execute = plan_execute
        self.search_engine = search_engine
        self.interval_seconds = interval_seconds

        logger.info("📋 Configuration summary:")
        logger.info(f"   • Query rewriting: {query_rewrite}")
        logger.info(f"   • Plan execution: {plan_execute}")
        logger.info(f"   • Search engine: {search_engine}")
        logger.info(f"   • Execution interval: {interval_seconds} seconds")

        logger.info("✅ BatchResponseGenerator successfully initialized")

    def deleteAzureAIAgent(self):
        """
        Delete the Azure AI agent if it exists
        This is useful for cleanup after batch evaluations
        """
        if hasattr(self.bing_grounding_search, "deleteAgent"):
            self.bing_grounding_search.deleteAgent()
            logger.info("✅ Deleted Azure AI agent")
        else:
            logger.warning("⚠️ No Azure AI agent to delete")

    async def generate_single_query(
        self,
        query_id: str,
        query_text: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout_seconds: int = 120,
        locale: Optional[str] = "en-US",
    ) -> Dict[str, Any]:
        """
        Evaluate a single query and return detailed results

        Args:
            query_id: Unique identifier for the query
            query_text: The user query text
            max_tokens: Maximum tokens for response generation
            temperature: Temperature for response generation
            timeout_seconds: Timeout for the evaluation

        Returns:
            Dictionary containing evaluation results
        """
        start_time = time.time()

        result = {
            "query_id": query_id,
            "query_text": query_text,
            "status": "unknown",
            "response_content": "",
            "response_time": 0.0,
            "error_message": None,
            "search_queries_count": 0,
            "streaming_chunks_count": 0,
        }

        try:
            logger.info(f"Generating query {query_id}: {query_text[:50]}...")

            # Create chat message
            messages = [ChatMessage(role="user", content=query_text)]

            # Generate response with timeout
            response_chunks = []
            search_queries_count = 0

            try:

                async def collect_response():
                    nonlocal search_queries_count
                    chunks = []
                    if self.search_engine == SearchEngine.BING_GROUNDING:
                        async for chunk in self.orchestrator.generate_response(
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            query_rewrite=self.query_rewrite,  # Disable for consistent evaluation
                            search_engine=self.search_engine,
                            search_crawler=self.search_crawler,
                            bing_grounding_search=self.bing_grounding_search,
                            stream=False,
                            elapsed_time=False,  # Disable elapsed time for evaluation,
                            locale=locale,
                        ):
                            chunks.append(chunk)
                            # Count search queries from streaming messages
                        return chunks

                    elif (
                        self.search_engine != SearchEngine.BING_GROUNDING
                        and self.plan_execute
                    ):
                        async for chunk in self.plan_executor.generate_response(
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            query_rewrite=self.query_rewrite,  # Disable for consistent evaluation
                            search_engine=self.search_engine,
                            search_crawler=self.search_crawler,
                            stream=False,
                            elapsed_time=False,  # Disable elapsed time for evaluation
                            locale=locale,
                        ):
                            chunks.append(chunk)
                            # Count search queries from streaming messages
                            if (
                                "Search plan established" in chunk
                                and "search queries" in chunk
                            ):
                                try:
                                    # Extract number from message like "2 search queries generated"
                                    parts = chunk.split("search queries")
                                    if parts:
                                        search_queries_count = int(parts[0].split()[-1])
                                except (ValueError, IndexError):
                                    pass
                        return chunks

                    else:
                        async for chunk in self.orchestrator.generate_response(
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            query_rewrite=self.query_rewrite,  # Disable for consistent evaluation
                            search_engine=self.search_engine,
                            search_crawler=self.search_crawler,
                            bing_grounding_search=self.bing_grounding_search,
                            stream=False,
                            elapsed_time=False,  # Disable elapsed time for evaluation
                            locale=locale,
                        ):
                            chunks.append(chunk)
                            # Count search queries from streaming messages
                        return chunks

                response_chunks = await asyncio.wait_for(
                    collect_response(), timeout=timeout_seconds
                )

            except asyncio.TimeoutError:
                raise Exception(f"Request timed out after {timeout_seconds} seconds")

            # Process response chunks
            response_content = ""
            streaming_progress_messages = 0

            for chunk in response_chunks:
                if chunk.startswith("data: ###"):
                    streaming_progress_messages += 1
                else:
                    response_content += chunk

            # Clean up response content
            response_content = response_content.strip()

            # Update result
            result.update(
                {
                    "status": "success",
                    "response_content": response_content,
                    "response_time": time.time() - start_time,
                    "search_queries_count": search_queries_count,
                    "streaming_chunks_count": len(response_chunks),
                    "streaming_progress_messages": streaming_progress_messages,
                }
            )

            logger.info(
                f"Query {query_id} completed successfully in {result['response_time']:.2f}s "
                f"(content length: {len(response_content)}, search queries: {search_queries_count})"
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Query {query_id} failed: {error_msg}")

            result.update(
                {
                    "status": "error",
                    "response_time": time.time() - start_time,
                    "error_message": error_msg,
                }
            )

        return result

    async def generate_batch(
        self,
        queries: List[Dict[str, str]],
        max_concurrent: int = 3,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        locale: Optional[str] = None,
        progress_interval: int = 10,
    ) -> Dict[str, Any]:
        """
        Generate a batch response of queries with concurrency control

        Args:
            queries: List of query dictionaries with 'id' and 'text' keys
            max_concurrent: Maximum concurrent requests
            max_tokens: Maximum tokens for response generation
            temperature: Temperature for response generation
            progress_interval: Progress reporting interval
            search_crawler: Search crawler instance to use
        Returns:
            Complete response results
        """
        response_results = []
        logger.info(
            f"Starting batch response generation of {len(queries)} queries with max_concurrent={max_concurrent}"
        )
        logger.info(
            "########### Current Response Generation Configuration: ############"
        )
        logger.info(
            f"max_concurrent: {max_concurrent}, max_tokens: {max_tokens}, temperature: {temperature}"
        )

        logger.info(
            f"query-rewrite: {self.query_rewrite}, plan-execute: {self.plan_execute}, search-engine: {self.search_engine}"
        )
        logger.info(f"Progress Interval: {progress_interval} queries")

        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_with_semaphore(query_data):
            await asyncio.sleep(self.interval_seconds)  # 실행 간격 추가
            async with semaphore:
                return await self.generate_single_query(
                    query_id=query_data["id"],
                    query_text=query_data["text"],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    locale=locale,
                )

        # Create tasks for all queries
        tasks = [generate_with_semaphore(query) for query in queries]

        # Process tasks with progress reporting
        completed = 0
        batch_start_time = time.time()

        for task in asyncio.as_completed(tasks):
            result = await task
            # self.metrics.add_result(result)
            response_results.append(result)
            completed += 1

            if completed % progress_interval == 0 or completed == len(queries):
                elapsed = time.time() - batch_start_time
                avg_time = elapsed / completed
                estimated_remaining = (len(queries) - completed) * avg_time

                logger.info(
                    f"Progress: {completed}/{len(queries)} "
                    f"({completed/len(queries)*100:.1f}%) - "
                    f"Elapsed: {elapsed:.1f}s, ETA: {estimated_remaining:.1f}s"
                )

        return {
            "generation_metadata": {
                "total_queries": len(queries),
                "max_concurrent": max_concurrent,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "generation_duration_seconds": time.time() - batch_start_time,
            },
            "response_results": response_results,
        }


def load_queries_from_csv(
    file_path: str, limit: Optional[int] = None
) -> List[Dict[str, str]]:
    """Load queries from CSV file"""
    queries = []

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for i, row in enumerate(reader):
                if limit is not None and i >= limit:
                    break

                # Handle different possible column names
                query_id = row.get("No", row.get("id", str(len(queries) + 1)))
                query_text = row.get("query", row.get("text", row.get("question", "")))

                if query_text.strip():
                    queries.append({"id": str(query_id), "text": query_text.strip()})

        logger.info(f"Loaded {len(queries)} queries from {file_path}")
        return queries

    except Exception as e:
        logger.error(f"Failed to load queries from {file_path}: {e}")
        raise


def save_results(results: Dict[str, Any], output_path: str) -> str:
    """Save response generation results to JSONL file"""
    try:
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as file:
            # Write each response result as a separate JSON line
            for result in results.get("response_results", []):
                jsonl_entry = {
                    "query": result.get("query_text", ""),
                    "response": result.get("response_content", ""),
                }
                file.write(json.dumps(jsonl_entry, ensure_ascii=False) + "\n")

        logger.info(f"Results saved to {output_path} in JSONL format")
        return output_path
    except Exception as e:
        logger.error(f"Failed to save results to {output_path}: {e}")
        raise


def generate_evaluation_report(data_file: str):

    # Load JSON file
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("rows", [])

    # Define score ranges
    score_range = [0, 1, 2, 3, 4, 5]
    freq_relevance = {score: 0 for score in score_range}

    # Count occurrences of each score
    for row in rows:
        relevance_score = row.get("outputs.relevance.relevance", 0)

        if relevance_score in freq_relevance:
            freq_relevance[relevance_score] += 1

    # Generate combined response chart
    def generate_response_chart():
        fig, ax = plt.subplots()
        x = np.arange(len(score_range))
        width = 0.3
        ax.bar(
            x - width,
            [freq_relevance[score] for score in score_range],
            width,
            label="Relevance",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(score_range)
        ax.set_xlabel("Score")
        ax.set_ylabel("Frequency")
        ax.set_title("Relevance Score")
        ax.legend()
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        chart_data = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return chart_data

    response_chart = generate_response_chart()

    # Generate HTML table
    table_html = '<table border="1" style="border-collapse: collapse;"><tr><th>Query</th><th>Response</th><th>Relevance</th></tr>'
    for row in rows:
        table_html += f"<tr><td>{row.get('inputs.query', '')}</td><td>{row.get('inputs.response', '')}</td><td>{row.get('outputs.relevance.relevance', '')}</td></tr>"
    table_html += "</table>"

    # Generate HTML content
    html_content = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Evaluation Results</title>
        <style>
            .image-container {{ display: flex; flex-wrap: wrap; justify-content: space-around; }}
            .image-container div {{ margin: 10px; text-align: center; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
        <script>
        function downloadCSV() {{
                var csv = 'Query,Response,Relevance\n';
                var rows = document.querySelectorAll('table tr');
                for (var i = 1; i < rows.length; i++) {{
                    var cols = rows[i].querySelectorAll('td');
                    var row = [];
                    for (var j = 0; j < cols.length; j++) {{
                        var cell = cols[j].innerText.replace(/"/g, '""');
                        row.push('"' + cell + '"');
                    }}
                    csv += row.join(',') + '\n';
                }}
                // BOM 추가
                var BOM = '\ufeff';
                var hiddenElement = document.createElement('a');
                hiddenElement.href = 'data:text/csv;charset=utf-8,' + encodeURIComponent(BOM + csv);
                hiddenElement.target = '_blank';
                hiddenElement.download = 'evaluation_results.csv';
                hiddenElement.click();
        }}
        </script>
    </head>
    <body>
        <h1>Evaluation Results</h1>
        <div class="image-container">
            <div><h2>Response Score Distribution</h2><img src="data:image/png;base64,{response_chart}"/></div>
        </div>
        <div style="display: flex; align-items: center;">
            <h2 style="margin-right: 16px;">Results Table</h2>
            <button onclick="downloadCSV()">Download CSV</button>
        </div>
        {table_html}
    </body>
    </html>
    """

    # Save HTML file with timestamp
    timezone = pytz.timezone("Asia/Seoul")
    timestamp = datetime.now(tz=timezone).strftime("%Y%m%d_%H%M%S")
    filename = f"evals/results/evaluation_report_{timestamp}.html"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML file '{filename}' generated.")


async def main():
    """Main evaluation function"""
    timezone = pytz.timezone("Asia/Seoul")
    current_date_time = datetime.now(tz=timezone).strftime("%Y-%m-%d_%H:%M:%S")

    parser = argparse.ArgumentParser(
        description="Batch evaluation for Microsoft product inquiry chatbot"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input CSV file containing queries"
    )
    parser.add_argument(
        "--gen_output",
        "-g",
        default=f"evals/results/generated_results_{current_date_time}.jsonl",
        help="Output JSON line file for generated results",
    )
    parser.add_argument(
        "--eval_output",
        "-e",
        default=f"evals/results/evaluation_results_{current_date_time}.json",
        help="Output JSON line file for evaluation results",
    )
    parser.add_argument(
        "--max_concurrent",
        "-c",
        type=int,
        default=1,
        help="Maximum number of concurrent requests",
    )
    parser.add_argument(
        "--max_tokens", "-t", type=int, help="Maximum tokens for response generation"
    )
    parser.add_argument(
        "--temperature", "-T", type=float, help="Temperature for response generation"
    )
    parser.add_argument(
        "--locale",
        "-lo",
        type=str,
        default="en-US",
        help="Locale for response generation",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=3,
        help="Limit on the number of queries to evaluate",
    )

    def str_to_bool(v):
        """Safely convert string to boolean"""
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser.add_argument(
        "--query_rewrite",
        "-q",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
        help="enable query rewriting (true/false)",
    )
    parser.add_argument(
        "--plan_execute",
        "-p",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
        help="enable plan execution (true/false)",
    )
    parser.add_argument(
        "--search_engine",
        "-s",
        type=str,
        default="grounding_bing",
        help="Search engine to use",
    )
    parser.add_argument(
        "--interval",
        "-it",
        type=float,
        default=1.0,
        help="Query execution interval (seconds)",
    )
    parser.add_argument(
        "--verbose", "-V", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # verbose 모드 처리
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("🔍 Detailed logging mode is enabled")

    try:
        # Load settings
        logger.info("⚙️ Loading Azure settings...")
        settings = Settings()
        logger.info("✅ Azure settings loaded successfully")

        # Load limited queries
        logger.info(f"📄 Loading queries from {args.input} (limit: {args.limit})...")
        queries = load_queries_from_csv(args.input, args.limit)
        logger.info(f"✅ {len(queries)} queries loaded")

        # Create evaluator
        logger.info("🔧 Creating Redis and search crawler...")
        redis_config = {
            "host": settings.REDIS_HOST,
            "port": settings.REDIS_PORT,
            "password": settings.REDIS_PASSWORD,
            "db": settings.REDIS_DB,
            "decode_responses": True,
        }

        # 검색 엔진별 크롤러 설정
        if args.search_engine == SearchEngine.BING_SEARCH_CRAWLING:
            search_crawler = BingSearchCrawler(redis_config=redis_config)
            logger.info("🔍 Bing Search Crawler is being used")
        elif args.search_engine == SearchEngine.BING_GROUNDING_CRAWLING:
            search_crawler = BingGroundingCrawler(redis_config=redis_config)
            logger.info("🔍 Bing Grounding Crawler is being used")
        elif args.search_engine == SearchEngine.BING_GROUNDING:
            search_crawler = BingSearchCrawler(redis_config=redis_config)
            bing_grounding_search = BingGroundingSearch(redis_config=redis_config)
            logger.info("🔍 Bing Grounding Search is being used")
        elif args.search_engine == SearchEngine.GOOGLE_SEARCH_CRAWLING:
            search_crawler = GoogleSearchCrawler(redis_config=redis_config)
            logger.info("🔍 Google Search Crawler is being used")
        else:
            search_crawler = BingSearchCrawler(redis_config=redis_config)
            logger.info("🔍 Defaulting to Bing Search Crawler")

        # FIX: Pass the correct boolean values to BatchResponseGenerator
        batch_response_generator = BatchResponseGenerator(
            settings=settings,
            search_crawler=search_crawler,
            bing_grounding_search=bing_grounding_search,
            search_engine=args.search_engine,
            query_rewrite=args.query_rewrite,
            plan_execute=args.plan_execute,
            interval_seconds=args.interval,
        )

        # Generate responses
        logger.info("🚀 Starting Azure-based batch response generation...")
        logger.info("📊 Execution parameters:")
        logger.info(f"   • Max concurrent requests: {args.max_concurrent}")
        logger.info(f"   • max_tokens: {args.max_tokens}")
        logger.info(f"   • temperature: {args.temperature}")
        logger.info(f"   • Execution interval: {args.interval} seconds")

        response_results = await batch_response_generator.generate_batch(
            queries=queries,
            max_concurrent=args.max_concurrent,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            locale=args.locale,
        )

        batch_response_generator.deleteAzureAIAgent()

        # Save results to jsonl
        logger.info("💾 Saving generation results...")
        gen_output_path = save_results(response_results, args.gen_output)

        # Run batch evaluation
        logger.info("🔍 Running batch evaluation...")
        batch_evaluator = BatchEvaluator(settings)

        batch_evaluator.batch_evaluate(
            eval_input_path=gen_output_path, eval_output_path=args.eval_output
        )

        # Generate summary report
        logger.info("📊 Generating evaluation report...")
        generate_evaluation_report(args.eval_output)

        logger.info("🎉 Batch evaluation completed successfully!")

    except Exception as e:
        logger.error(f"❌ Batch evaluation failed: {e}")
        logger.exception("Detailed error information:")
        raise


if __name__ == "__main__":
    asyncio.run(main())
