# Multi-Agent Document Research Chatbot

## Overview
For a walkthrough, check out the video overview here: xxx.mp4

![Architecture Diagram](images/multi-agent-doc-research-architecture-Page-2.png)

This AI-powered chatbot performs custom deep research on uploaded documents using a semantic chunking strategy for precise and meaningful vectorization. Through multi-agent collaboration, it delivers accurate, context-aware answers to user queries.

Built with FastAPI, Azure OpenAI, and Chainlit, the system showcases advanced techniques for enhancing LLM-based applications—such as agentic patterns, modular architecture, multi-agent orchestration, and evaluation support.

At its core, the multi-agent deep research engine combines Microsoft Agent Framework and Semantic Kernel to generate high-quality analytical reports. By employing group chat coordination and a magnetic multi-agent pattern, it achieves deeper reasoning and consistent, well-structured outputs.

## Key features
E2E sample, Multi-Agent orchestration, Group Chat Pattern, Magentic Pattern, MS Agent Framework, Semantic Kernel, AI Search, Semantic chunking, Document Intelligence, Modular architecture

## Multi-Agent Implementation
- MS Agent Framework GroupChat: Group chat coordination using MS Agent Framework. (members: WriterAgent, ReviwerAgent)
- MS Agent Framework Magentic: Basic Magnetic pattern for deep reasoning and iterative refinement. (members: MagenticManagerAgent, ResearchAnalystAgent, ResearchWriterAgent, ResearchReviewerAgent)
- Semantic Kernel GroupChat: Group chat coordination using Semantic Kernel. (members: WriterAgent, ReviwerAgent)
- Semantic Kernel Magentic(Deep-Research-Agents): Deep research agents using Semantic Kernel. (members: MagenticManagerAgent, LeadResearchAgent, CredibilityCriticAgent, CitationAgent, ReportWriterAgent)
- Vanilla AOAI SDK: Standard SDK based custom workflow implementation using parallel calls.(members: WrtiterAgent, ReviewerAgent)

## Azure AI Services
AOAI, Azure AI Foundry, Azure AI Search

