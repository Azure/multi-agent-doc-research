"""
Unified File Upload Plugin

This plugin handles the complete file upload workflow:
- File upload request handling
- Document processing and chunking with multiple strategies (semantic-aware, page-based)
- Vector storage in Azure AI Search
- Upload status tracking and notification

All functionalities are consolidated into a single, simple plugin.
"""

import json
import logging
import hashlib
import os
import re
from typing import Dict, List
from pathlib import Path

from semantic_kernel.functions import kernel_function
from config.config import Settings
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentContentFormat, AnalyzeDocumentRequest, AnalyzeResult
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import AsyncAzureOpenAI
from utils.token_counter import get_token_numbers

logger = logging.getLogger(__name__)


class SemanticAwareTextSplitter:
    """
    Enhanced text splitter that creates chunks based on semantic sections
    and maintains target token count (400±100 tokens) with overlap.
    """
    
    def __init__(self, 
                 target_tokens: int = 2000,
                 token_variance: int = 200,
                 overlap_percentage: float = 0.1,
                 model_name: str = "gpt-4"):
        self.target_tokens = target_tokens
        self.min_tokens = target_tokens - token_variance
        self.max_tokens = target_tokens + token_variance
        self.overlap_percentage = overlap_percentage
        self.model_name = model_name
        
        # Semantic separators prioritized by meaning preservation
        self.semantic_separators = [
            # Document structure markers
            "\n\n# ",      # Major sections (H1)
            "\n\n## ",     # Sub-sections (H2)
            "\n\n### ",    # Sub-sub-sections (H3)
            "\n\n#### ",   # Minor sections (H4)
            
            # Content boundaries
            "\n\n---\n",   # Horizontal rules with newlines
            "\n\n***\n",   # Alternative horizontal rules
            "\n\n___\n",   # Another horizontal rule variant
            "\n---\n",     # Simple horizontal rules
            
            # Logical breaks
            "\n\n\n\n",    # Quad line breaks
            "\n\n\n",      # Triple line breaks
            "\n\n",        # Double line breaks (paragraph boundaries)
            
            # Sentence boundaries with context
            ".\n\n",       # Sentence end with paragraph break
            ". \n\n",      # Sentence end with space and paragraph break
            ".\n",         # Sentence end with newline
            ". \n",        # Sentence end with space and newline
            ". ",          # Simple sentence boundaries
            "! ",          # Exclamation boundaries
            "? ",          # Question boundaries
            
            # List and enumeration patterns
            "\n\n- ",      # Bullet points with paragraph break
            "\n\n* ",      # Alternative bullet points
            "\n\n+ ",      # Plus bullet points
            "\n- ",        # Simple bullet points
            "\n* ",        # Simple alternative bullet points
            "\n+ ",        # Simple plus bullet points
            "\n\n1. ",     # Numbered lists with paragraph break
            "\n\n2. ",     # Continue numbered pattern
            "\n1. ",       # Simple numbered lists
            "\n2. ",       # Simple continue numbered
            "\n\t",        # Tab-indented content
            
            # Final fallbacks (더 안전하게)
            "\n",          # Any line break
            ". ",          # Period with space (repeated for emphasis)
            " ",           # Word boundaries
        ]
    
    def split_text_with_document_intelligence(self, result) -> List[str]:
        """
        Document Intelligence 결과를 마크다운 헤더(#, ##, ###) 기준으로 큰 단위 섹션별로 분할
        """
        content = result.content if result.content else ""
        
        if not content:
            logger.warning("No content found in Document Intelligence result")
            return []
        
        # 마크다운 헤더로 섹션 분할
        chunks = self._split_by_markdown_headers(content)
        
        logger.info(f"Created {len(chunks)} chunks based on markdown headers")
        return chunks

    def _split_by_markdown_headers(self, content: str) -> List[str]:
        """
        마크다운 헤더(#, ##) 기준으로 스마트 섹션 분할
        - H1(#) 하위에 H2(##)가 2개 이상 있으면 각 H2를 별도 청크로 분할 (H1 제목 포함)
        - H1 하위에 H2가 1개뿐이면 하나의 청크로 유지
        - 토큰 수 제한도 고려
        """
        if not content:
            return []
        
        lines = content.split('\n')
        sections = []
        current_section = []
        current_h1 = None
        
        # 1단계: 라인별로 구조 파악
        for line in lines:
            stripped_line = line.strip()
            
            if re.match(r'^#\s+', stripped_line):  # H1 헤더
                # 이전 섹션 저장
                if current_section:
                    sections.append({
                        'h1': current_h1,
                        'content': current_section
                    })
                
                # 새로운 H1 섹션 시작
                current_h1 = line
                current_section = [line]
                
            elif re.match(r'^##\s+', stripped_line):  # H2 헤더
                if current_h1:  # H1이 있는 경우에만
                    current_section.append(line)
                else:
                    # H1 없이 H2가 나온 경우 (독립적인 섹션으로 처리)
                    if current_section:
                        sections.append({
                            'h1': None,
                            'content': current_section
                        })
                    current_section = [line]
            else:
                # 일반 컨텐츠 라인
                current_section.append(line)
        
        # 마지막 섹션 저장
        if current_section:
            sections.append({
                'h1': current_h1,
                'content': current_section
            })
        
        # 2단계: 각 섹션을 스마트하게 청크로 분할
        chunks = []
        
        for section in sections:
            h1_header = section['h1']
            content_lines = section['content']
            
            if not h1_header:
                # H1이 없는 경우 (독립 섹션)
                chunk_text = '\n'.join(content_lines).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                continue
            
            # H1이 있는 경우 - H2 서브섹션 확인
            h2_positions = []
            for i, line in enumerate(content_lines):
                if re.match(r'^##\s+', line.strip()):
                    h2_positions.append(i)
            
            if len(h2_positions) <= 1:
                # H2가 1개 이하인 경우 - 전체를 하나의 청크로
                chunk_text = '\n'.join(content_lines).strip()
                if chunk_text:
                    # 토큰 수 체크
                    try:
                        token_count = get_token_numbers(chunk_text, "gpt-4")
                        if token_count <= self.max_tokens:
                            chunks.append(chunk_text)
                        else:
                            # 토큰이 너무 많으면 H2별로 강제 분할
                            if h2_positions:
                                sub_chunks = self._split_h1_section_by_h2(h1_header, content_lines, h2_positions)
                                chunks.extend(sub_chunks)
                            else:
                                chunks.append(chunk_text)  # H2가 없으면 그대로 추가
                    except:
                        chunks.append(chunk_text)  # 토큰 계산 실패 시 그대로 추가
            else:
                # H2가 2개 이상인 경우 - H2별로 분할 (H1 제목 포함)
                sub_chunks = self._split_h1_section_by_h2(h1_header, content_lines, h2_positions)
                chunks.extend(sub_chunks)
        
        # 빈 청크 제거
        final_chunks = [chunk for chunk in chunks if chunk.strip()]
        
        if not final_chunks and content.strip():
            logger.info("No markdown headers found, treating entire content as one chunk")
            final_chunks = [content.strip()]
        
        logger.info(f"Split content into {len(final_chunks)} smart header-based chunks")
        return final_chunks

    def _split_h1_section_by_h2(self, h1_header: str, content_lines: List[str], h2_positions: List[int]) -> List[str]:
        """
        H1 섹션을 H2별로 분할하되, 각 H2 청크에 H1 제목을 포함
        """
        chunks = []
        
        for i, h2_pos in enumerate(h2_positions):
            # 다음 H2 위치 또는 끝까지
            next_h2_pos = h2_positions[i + 1] if i + 1 < len(h2_positions) else len(content_lines)
            
            # H2 섹션 내용 추출
            h2_section_lines = content_lines[h2_pos:next_h2_pos]
            
            # H1 + H2 섹션 조합
            chunk_lines = [h1_header] + h2_section_lines
            chunk_text = '\n'.join(chunk_lines).strip()
            
            if chunk_text:
                chunks.append(chunk_text)
        
        return chunks
    
    def split_text_with_semantic_awareness(self, text: str) -> List[str]:
        """
        Split text into semantically meaningful chunks with target token count.
        """
        if not text or not text.strip():
            logger.warning("Empty or whitespace-only text provided for splitting")
            return []
        
        # Clean and normalize text first
        text = self._clean_text(text)
        
        # Log the text format for debugging
        logger.info(f"Text format analysis - Length: {len(text)}, First 200 chars: {repr(text[:200])}")
        
        try:
            # First, try to identify major sections using headers and structure
            sections = self._identify_semantic_sections(text)
            
            if not sections:
                logger.warning("No sections identified, using fallback chunking")
                return self._fallback_chunking(text)
            
            chunks = []
            for section in sections:
                if section and section.strip():  # Ensure section is not empty
                    section_chunks = self._split_section_to_target_tokens(section)
                    chunks.extend(section_chunks)
            
            if not chunks:
                logger.warning("No chunks created from sections, using fallback")
                return self._fallback_chunking(text)
            
            # Apply overlap between chunks
            overlapped_chunks = self._apply_overlap(chunks)
            
            return overlapped_chunks
            
        except Exception as e:
            logger.error(f"Error in semantic splitting: {e}")
            logger.info("Falling back to simple chunking")
            return self._fallback_chunking(text)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text from Document Intelligence.
        Enhanced for better text matching during page number detection.
        """
        if not text:
            return ""
        
        # 1. 기본 공백 정규화
        text = re.sub(r'\n{4,}', '\n\n\n', text)  # 연속된 줄바꿈 제한
        text = re.sub(r' {3,}', '  ', text)       # 연속된 공백 제한
        text = re.sub(r'\t+', ' ', text)          # 탭을 공백으로 변환
    
        # 2. 마크다운 패턴 정규화 (Document Intelligence 출력 특성 반영)
        text = re.sub(r'\*{3,}', '***', text)     # 연속된 asterisk 정규화
        text = re.sub(r'-{4,}', '---', text)      # 연속된 dash 정규화  
        text = re.sub(r'_{4,}', '___', text)      # 연속된 underscore 정규화
    
        # 3. 마크다운 헤더 정규화 (페이지 번호 검색을 위해 중요)
        # 헤더 앞뒤 공백 정규화
        text = re.sub(r'\n\s*#+\s*', '\n# ', text)      # H1 헤더 정규화
        text = re.sub(r'\n\s*##\s*', '\n## ', text)     # H2 헤더 정규화  
        text = re.sub(r'\n\s*###\s*', '\n### ', text)   # H3 헤더 정규화
        text = re.sub(r'\n\s*####\s*', '\n#### ', text) # H4 헤더 정규화
    
        # 4. 특수 문자 및 인코딩 문제 정규화
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)      # 비ASCII 문자를 공백으로 (선택적)
        text = re.sub(r'[""''„]', '"', text)          # 다양한 따옴표를 표준 따옴표로
        text = re.sub(r'[–—]', '-', text)               # 다양한 대시를 표준 하이픈으로
        text = re.sub(r'…', '...', text)                # 말줄임표 정규화
    
        # 5. 리스트 패턴 정규화
        text = re.sub(r'\n\s*[•·▪▫‣⁃]\s*', '\n- ', text)  # 다양한 불릿을 표준 하이픈으로
        text = re.sub(r'\n\s*\d+\.\s*', lambda m: f'\n{m.group().strip()} ', text)  # 번호 리스트 정규화
    
        # 6. 표와 구조화된 데이터 정규화
        text = re.sub(r'\|\s*\|', '| |', text)          # 빈 테이블 셀 정규화
        text = re.sub(r'\|\s+', '| ', text)             # 테이블 구분자 후 공백 정규화
        text = re.sub(r'\s+\|', ' |', text)             # 테이블 구분자 앞 공백 정규화
    
        # 7. 문장 경계 정규화 (검색 향상을 위해)
        text = re.sub(r'([.!?])\s*\n\s*([A-Z])', r'\1 \2', text)  # 문장 끝과 다음 문장 시작 정규화
        text = re.sub(r'([.!?])\s{2,}', r'\1 ', text)             # 문장 끝 후 과도한 공백 제거
    
        # 8. 최종 정리
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)   # 과도한 줄바꿈 정리
        text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)  # 줄 끝 공백 제거
        text = re.sub(r'^[ \t]+', '', text, flags=re.MULTILINE)  # 줄 시작 공백 제거 (헤더 제외)
    
        # 9. 검색용 텍스트 정규화 함수들을 위한 기본 처리
        # (나중에 _get_chunk_page_number에서 사용할 정규화된 텍스트)
        normalized_text = text.strip()
    
        # 10. 로깅 (디버깅용)
        if len(text) != len(normalized_text):
            logger.debug(f"Text normalization: {len(text)} -> {len(normalized_text)} characters")
    
        return normalized_text
    
    def _normalize_text_for_search(self, text: str) -> str:
        """
        텍스트를 검색에 최적화된 형태로 정규화
        페이지 번호 검색 시 매칭 성공률 향상을 위해 사용
        """
        if not text:
            return ""
        
        # 1. 기본 정규화
        normalized = text.lower().strip()
        
        # 2. 마크다운 헤더 제거 (검색 시 방해가 될 수 있음)
        normalized = re.sub(r'^#+\s*', '', normalized, flags=re.MULTILINE)
        
        # 3. 특수문자와 구두점 정규화
        normalized = re.sub(r'[^\w\s가-힣]', ' ', normalized)  # 영문, 숫자, 한글, 공백만 유지
        
        # 4. 공백 정규화
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # 5. 앞뒤 공백 제거
        normalized = normalized.strip()
        
        return normalized
    
    def _extract_search_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """
        텍스트에서 검색에 유용한 키워드들을 추출
        페이지 번호 검색 시 fallback으로 사용
        """
        if not text:
            return []
        
        # 정규화된 텍스트에서 키워드 추출
        normalized = self._normalize_text_for_search(text)
        words = normalized.split()
        
        # 1. 의미있는 단어들 필터링 (길이 3 이상)
        meaningful_words = [word for word in words if len(word) >= 3]
        
        # 2. 일반적인 불용어 제거 (한영 혼합)
        stopwords = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'she', 'use', 'way', 'will', 'yet',
            '이', '그', '저', '것', '수', '등', '및', '또는', '하지만', '그러나', '따라서', '그리고', '있다', '없다', '이며', '에서', '으로', '에게', '에서도', '또한'
        }
        
        filtered_words = [word for word in meaningful_words if word not in stopwords]
        
        # 3. 빈도순으로 정렬하되, 첫 번째 등장하는 순서도 고려
        word_positions = {word: idx for idx, word in enumerate(filtered_words)}
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1
    
        # 4. 빈도와 위치를 종합하여 점수 계산
        word_scores = {}
        for word, count in word_counts.items():
            position_score = 1.0 / (word_positions[word] + 1)  # 앞쪽에 나올수록 높은 점수
            frequency_score = count
            word_scores[word] = frequency_score + position_score
    
        # 5. 점수순으로 정렬하여 상위 키워드 반환
        sorted_keywords = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [keyword for keyword, _ in sorted_keywords[:max_keywords]]
    
    def _identify_semantic_sections(self, text: str) -> List[str]:
        """
        Identify semantic sections based on document structure.
        Enhanced to handle Document Intelligence output better.
        """
        # Look for clear section markers (more flexible patterns)
        section_patterns = [
            r'\n\n#+\s+[^\n]+\n',           # Markdown headers
            r'\n\n\d+\.\s+[^\n]+\n',        # Numbered sections  
            r'\n\n[A-Z][A-Z\s]{2,}[A-Z]\n', # ALL CAPS headers (more specific)
            r'\n\n[^\n]+:\s*\n',            # Colon-ended headers
            r'\n\n[IVX]+\.\s+[^\n]+\n',     # Roman numeral sections
            r'\n\n[A-Z]\.\s+[^\n]+\n',      # Letter sections (A., B., etc.)
        ]
        
        # Find all potential section breaks
        breaks = [0]  # Always start at beginning
        
        for pattern in section_patterns:
            try:
                matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
                for match in matches:
                    break_pos = match.start()
                    if break_pos not in breaks and break_pos > 0:
                        breaks.append(break_pos)
            except re.error as e:
                logger.warning(f"Regex pattern error: {e}")
                continue
        
        breaks.append(len(text))  # Always end at text end
        breaks = sorted(list(set(breaks)))  # Remove duplicates and sort
        
        # Create sections
        sections = []
        for i in range(len(breaks) - 1):
            start_pos = breaks[i]
            end_pos = breaks[i + 1]
            section = text[start_pos:end_pos].strip()
            
            if section and len(section) > 30:  # Skip very short sections
                sections.append(section)
        
        # If no clear sections found, split by double line breaks
        if len(sections) <= 1:
            logger.info("No clear sections found with patterns, trying paragraph splits")
            paragraph_sections = []
            for para in text.split('\n\n'):
                para = para.strip()
                if para and len(para) > 30:
                    paragraph_sections.append(para)
            
            if paragraph_sections:
                sections = paragraph_sections
            else:
                # Last resort: return the entire text
                logger.info("No paragraph sections found, using entire text")
                sections = [text] if text and text.strip() else []
        
        logger.info(f"Identified {len(sections)} sections")
        return sections
    
    def _split_section_to_target_tokens(self, section: str) -> List[str]:
        """
        Split a section into chunks that meet target token requirements.
        """
        if not section or not section.strip():
            return []
        
        try:
            current_tokens = get_token_numbers(section, self.model_name)
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            # Fallback: estimate tokens based on character count
            current_tokens = len(section) // 4  # Rough estimate
        
        # If section is within target range, return as is
        if self.min_tokens <= current_tokens <= self.max_tokens:
            return [section]
        
        # If section is too small, return as is (will be handled by overlap logic)
        if current_tokens < self.min_tokens:
            return [section]
        
        # If section is too large, split it recursively
        return self._recursive_split(section)
    
    def _recursive_split(self, text: str) -> List[str]:
        """
        Recursively split text using semantic separators until target token count is reached.
        Enhanced error handling for empty separators.
        """
        if not text or not text.strip():
            return []
        
        try:
            current_tokens = get_token_numbers(text, self.model_name)
        except Exception as e:
            logger.error(f"Error counting tokens in recursive split: {e}")
            current_tokens = len(text) // 4  # Fallback estimation
        
        if current_tokens <= self.max_tokens:
            return [text]
        
        # Try each separator in order of semantic importance
        for separator in self.semantic_separators:
            if not separator:  # Skip empty separators
                logger.warning("Encountered empty separator, skipping")
                continue
                
            if separator in text and len(text.split(separator)) > 1:
                try:
                    parts = text.split(separator, 1)
                    if len(parts) != 2:
                        continue
                        
                    left_part = parts[0].strip()
                    right_part_content = parts[1].strip()
                    
                    # Skip if either part is empty
                    if not left_part or not right_part_content:
                        continue
                    
                    # Reconstruct right part with separator if it's a meaningful separator
                    if separator.strip():  # If separator has meaningful content
                        right_part = separator + right_part_content
                    else:
                        right_part = right_part_content
                    
                    # Check if split produces reasonable chunks
                    try:
                        left_tokens = get_token_numbers(left_part, self.model_name)
                        right_tokens = get_token_numbers(right_part, self.model_name)
                    except Exception as e:
                        logger.error(f"Error counting tokens for split parts: {e}")
                        continue
                    
                    # If left part is in acceptable range, process both parts
                    if self.min_tokens <= left_tokens <= self.max_tokens:
                        result = [left_part]
                        result.extend(self._recursive_split(right_part))
                        return [chunk for chunk in result if chunk and chunk.strip()]
                    
                    # If left part is still too big, continue splitting it
                    elif left_tokens > self.max_tokens:
                        left_result = self._recursive_split(left_part)
                        right_result = self._recursive_split(right_part)
                        result = left_result + right_result
                        return [chunk for chunk in result if chunk and chunk.strip()]
                        
                except Exception as e:
                    logger.error(f"Error processing separator '{separator}': {e}")
                    continue
        
        # Fallback: use character-based splitting if no good separator found
        logger.warning("No suitable separators found, using character-based splitting")
        return self._character_split(text)
    
    def _character_split(self, text: str) -> List[str]:
        """
        Last resort: split by characters while trying to preserve word boundaries.
        """
        if not text or not text.strip():
            return []
        
        # Calculate approximate character count for target tokens
        # Rough estimation: 1 token ≈ 4 characters for English text
        target_chars = self.target_tokens * 4
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + target_chars, len(text))
            
            # Try to end at a word boundary
            if end < len(text):
                # Look backwards for a space or punctuation
                for i in range(end, max(start + target_chars // 2, start), -1):
                    if text[i] in ' .,;:!?\n':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk and len(chunk) > 10:  # Only add non-trivial chunks
                chunks.append(chunk)
            
            start = end
        
        return chunks
    
    
    
    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """
        Apply overlap between consecutive chunks.
        """
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if not chunk or not chunk.strip():
                continue
                
            if i == 0:
                # First chunk: no prefix overlap needed
                overlapped_chunk = chunk
            else:
                # Calculate overlap with previous chunk
                prev_chunk = chunks[i-1]
                if not prev_chunk or not prev_chunk.strip():
                    overlapped_chunk = chunk
                else:
                    overlap_chars = int(len(prev_chunk) * self.overlap_percentage)
                    
                    if overlap_chars > 0:
                        # Get overlap from end of previous chunk
                        overlap_text = prev_chunk[-overlap_chars:].strip()
                        
                        # Find a good breaking point in the overlap
                        sentences = re.split(r'[.!?]+\s+', overlap_text)
                        if len(sentences) > 1:
                            overlap_text = sentences[-1]  # Take the last complete sentence
                        
                        overlapped_chunk = overlap_text + "\n\n" + chunk
                    else:
                        overlapped_chunk = chunk
            
            # Ensure the final chunk doesn't exceed token limits
            try:
                chunk_tokens = get_token_numbers(overlapped_chunk, self.model_name)
                if chunk_tokens > self.max_tokens:
                    # If overlap made it too big, use original chunk
                    overlapped_chunk = chunk
            except Exception as e:
                logger.error(f"Error checking token count for overlapped chunk: {e}")
                overlapped_chunk = chunk
            
            if overlapped_chunk and overlapped_chunk.strip():
                overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks


class PageBasedTextSplitter:
    """
    Page-based text splitter that chunks documents by pages with overlap.
    """
    
    def __init__(self, overlap_percentage: float = 0.1, model_name: str = "gpt-4"):
        self.overlap_percentage = overlap_percentage
        self.model_name = model_name
    
    def split_text_by_pages(self, result) -> List[Dict]:
        """
        Split document into page-based chunks with overlap.
        
        Args:
            result: Document Intelligence analysis result
            
        Returns:
            List of page-based chunks with metadata
        """
        if not result.pages:
            return []
        
        page_chunks = []
        
        for page_idx, page in enumerate(result.pages):
            page_number = page.page_number
            
            # Extract content for this specific page
            page_content = self._extract_page_content(result, page_number)
            
            if not page_content or len(page_content.strip()) < 50:
                continue
            
            # Apply overlap with previous page if not the first page
            if page_idx > 0 and page_chunks:
                prev_page_content = page_chunks[-1]["raw_content"]
                page_content_with_overlap = self._apply_page_overlap(prev_page_content, page_content)
            else:
                page_content_with_overlap = page_content
            
            # Count tokens for this page
            page_tokens = get_token_numbers(page_content_with_overlap, self.model_name)
            
            chunk_data = {
                "content": page_content_with_overlap,
                "raw_content": page_content,  # Store original content for next page overlap
                "page_number": page_number,
                "tokens": page_tokens,
                "length": len(page_content_with_overlap)
            }
            
            page_chunks.append(chunk_data)
        
        return page_chunks
    
    def _extract_page_content(self, result, target_page_number: int) -> str:
        """
        Extract content for a specific page from Document Intelligence result.
        """
        page_content_parts = []
        
        # Extract paragraphs for this page
        if result.paragraphs:
            for paragraph in result.paragraphs:
                if paragraph.bounding_regions:
                    for region in paragraph.bounding_regions:
                        if region.page_number == target_page_number:
                            page_content_parts.append(paragraph.content)
                            break

        
        
        # If no paragraphs found, try to extract from tables for this page
        if not page_content_parts and result.tables:
            for table in result.tables:
                if table.bounding_regions:
                    for region in table.bounding_regions:
                        if region.page_number == target_page_number:
                            # Extract table content
                            table_content = self._extract_table_content(table)
                            if table_content:
                                page_content_parts.append(table_content)
                            break
        
        return "\n\n".join(page_content_parts)
    
    def _extract_table_content(self, table) -> str:
        """
        Extract content from a table structure.
        """
        if not table.cells:
            return ""
        
        # Simple table extraction - can be enhanced
        table_text = []
        current_row = -1
        row_content = []
        
        for cell in table.cells:
            if cell.row_index != current_row:
                if row_content:
                    table_text.append(" | ".join(row_content))
                current_row = cell.row_index
                row_content = []
            row_content.append(cell.content or "")
        
        if row_content:
            table_text.append(" | ".join(row_content))
        
        return "\n".join(table_text)
    
    def _apply_page_overlap(self, prev_page_content: str, current_page_content: str) -> str:
        """
        Apply overlap between consecutive pages.
        """
        if not prev_page_content:
            return current_page_content
        
        # Calculate overlap size
        overlap_chars = int(len(prev_page_content) * self.overlap_percentage)
        
        if overlap_chars > 0:
            # Get overlap from end of previous page
            overlap_text = prev_page_content[-overlap_chars:].strip()
            
            # Find a good breaking point (end of sentence or paragraph)
            sentences = re.split(r'[.!?]+\s+', overlap_text)
            if len(sentences) > 1:
                overlap_text = ". ".join(sentences[-2:]) + "."  # Take last complete sentences
            
            # Combine with current page content
            return overlap_text + "\n\n" + current_page_content
        
        return current_page_content


class UnifiedFileUploadPlugin:
    """Unified plugin for file upload and document processing with multiple chunking strategies."""
    
    def __init__(self):
        """Initialize the UnifiedFileUploadPlugin with Azure services."""
        self.settings = Settings()
        
        # Get processing method from environment variable
        self.processing_method = os.getenv("PROCESSING_METHOD", "semantic").lower()
        logger.info(f"Using processing method: {self.processing_method}")
        
        # Azure AI Search
        self.search_client = SearchClient(
            endpoint=self.settings.AZURE_AI_SEARCH_ENDPOINT,
            index_name=self.settings.AZURE_AI_SEARCH_INDEX_NAME,
            credential=AzureKeyCredential(self.settings.AZURE_AI_SEARCH_API_KEY)
        )
        
        # Document Intelligence
        self.doc_intelligence_client = DocumentIntelligenceClient(
            endpoint=self.settings.AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,
            credential=AzureKeyCredential(self.settings.AZURE_DOCUMENT_INTELLIGENCE_API_KEY)
        )
        
        # OpenAI for embeddings
        self.openai_client = AsyncAzureOpenAI(
            api_key=self.settings.AZURE_OPENAI_API_KEY,
            api_version=self.settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=self.settings.AZURE_OPENAI_ENDPOINT
        )
        
        # Initialize text splitters based on processing method
        if self.processing_method == "semantic":
            self.semantic_splitter = SemanticAwareTextSplitter(
                target_tokens=400,
                token_variance=100,
                overlap_percentage=0.1,
                model_name="gpt-4"
            )
        elif self.processing_method == "page":
            self.page_splitter = PageBasedTextSplitter(
                overlap_percentage=0.1,
                model_name="gpt-4"
            )
        else:
            # Default to semantic if unknown method
            logger.warning(f"Unknown processing method: {self.processing_method}. Defaulting to semantic.")
            self.processing_method = "semantic"
            self.semantic_splitter = SemanticAwareTextSplitter(
                target_tokens=400,
                token_variance=100,
                overlap_percentage=0.1,
                model_name="gpt-4"
            )
        
    async def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Azure OpenAI."""
        try:
            response = await self.openai_client.embeddings.create(
                input=text,
                model=self.settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def _generate_doc_hash(self, file_path: str) -> str:
        """Generate MD5 hash of file content for duplicate detection."""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error generating file hash: {e}")
            return ""
    
    async def _file_exists_in_vector_db(self, file_hash: str) -> bool:
        """Check if file already exists in vector database."""
        try:
            search_results = self.search_client.search(
                search_text="*",
                filter=f"file_hash eq '{file_hash}'",
                top=1
            )
            
            for _ in search_results:
                return True
            return False
        except Exception as e:
            logger.error(f"Error checking file existence: {e}")
            return False
    
    async def _process_pdf_with_semantic_chunking(
        self, 
        file_path: str, 
        original_filename: str,
        document_type: str,
        company: str,
        industry: str,
        report_year: str,
        result,
        is_debug: bool = False
    ) -> List[Dict]:
        """Process PDF using semantic-aware chunking."""
        try:
            # Extract content and metadata
            content = result.content if result.content else ""
            
            page_count = len(result.pages) if result.pages else 0
            
            logger.info(f"Processing with semantic chunking: {page_count} pages, {len(content)} characters")
            
            # Generate file hash for duplicate detection
            file_hash = self._generate_doc_hash(file_path)
            file_name = original_filename
            
            documents = []
            
            # Use markdown header-based chunking (simple and effective)
            logger.info("Starting markdown header-based chunking...")
            semantic_chunks = self.semantic_splitter.split_text_with_document_intelligence(result)
            
            logger.info(f"Created {len(semantic_chunks)} header-based chunks")
            
            # Process each chunk
            for chunk_num, chunk_text in enumerate(semantic_chunks, 1):
                chunk_text = chunk_text.strip()

                if is_debug:
                    # Debug: Save chunk text to file for analysis
                    debug_dir = Path("debug_chunks")
                    debug_dir.mkdir(exist_ok=True)
                    debug_file = debug_dir / f"{file_hash}_header_chunk_{chunk_num}.txt"
                    with open(debug_file, "w", encoding="utf-8") as f:
                        f.write(f"=== CHUNK {chunk_num} DEBUG INFO ===\n")
                        f.write(f"File: {original_filename}\n")
                        f.write(f"Chunk Number: {chunk_num}\n")
                        f.write(f"Text Length: {len(chunk_text)}\n")
                        f.write(f"Processing Method: markdown_header_based\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(chunk_text)
                    logger.info(f"Debug: Saved chunk {chunk_num} to {debug_file}")

                try:
                    # Count tokens for this chunk
                    chunk_tokens = get_token_numbers(chunk_text, "gpt-4")
                    
                    # Generate embedding
                    embedding = await self._get_embedding(chunk_text)
                    
                    # Create document object for vector storage
                    doc_id = f"{file_hash}_semantic_{chunk_num}"
                    
                    # Determine page number using Document Intelligence paragraph information
                    page_number = self._get_chunk_page_number(chunk_text, result)
                    
                    document = {
                        "docId": doc_id,
                        "content": chunk_text,
                        "content_vector": embedding,
                        "title": f"{file_name} - Section {chunk_num}",
                        "file_name": file_name,
                        "file_hash": file_hash,
                        "page_number": page_number,
                        "paragraph_number": chunk_num,
                        "chunk_number": chunk_num,
                        "document_type": document_type,
                        "company": company,
                        "industry": industry,
                        "report_year": report_year,
                        "source": file_path,
                        "metadata": json.dumps({
                            "total_pages": page_count,
                            "total_chunks": len(semantic_chunks),
                            "chunk_tokens": chunk_tokens,
                            "chunk_length": len(chunk_text),
                            "file_size": Path(file_path).stat().st_size,
                            "processing_method": "markdown_header_based_chunking",
                            "min_chunk_size": 100,
                            "header_based_splitting": True
                        })
                    }
                    
                    documents.append(document)
                    logger.info(f"Processed header-based chunk {chunk_num}: {chunk_tokens} tokens, {len(chunk_text)} characters")
                
                except Exception as e:
                    logger.error(f"Error processing semantic chunk {chunk_num}: {e}")
                    continue
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in semantic chunking: {e}")
            return []
    
    async def _process_pdf_with_page_chunking(
        self, 
        file_path: str, 
        original_filename: str,
        document_type: str,
        company: str,
        industry: str,
        report_year: str,
        result,
        is_debug: bool = False
    ) -> List[Dict]:
        """Process PDF using page-based chunking."""
        try:
            page_count = len(result.pages) if result.pages else 0
            logger.info(f"Processing with page chunking: {page_count} pages")
            
            # Generate file hash for duplicate detection
            file_hash = self._generate_doc_hash(file_path)
            file_name = original_filename
            
            documents = []
            
            # Use page-based chunking
            logger.info("Starting page-based chunking...")
            page_chunks = self.page_splitter.split_text_by_pages(result)
            
            logger.info(f"Created {len(page_chunks)} page chunks")
            
            # Process each page chunk
            for chunk_num, page_chunk in enumerate(page_chunks, 1):
                chunk_text = page_chunk["content"].strip()

                if is_debug:
                    # Debug: Save chunk text to file for analysis
                    debug_dir = Path("debug_chunks")
                    debug_dir.mkdir(exist_ok=True)
                    debug_file = debug_dir / f"{file_hash}_header_chunk_{chunk_num}.txt"
                    with open(debug_file, "w", encoding="utf-8") as f:
                        f.write(f"=== CHUNK {chunk_num} DEBUG INFO ===\n")
                        f.write(f"File: {original_filename}\n")
                        f.write(f"Chunk Number: {chunk_num}\n")
                        f.write(f"Text Length: {len(chunk_text)}\n")
                        f.write(f"Processing Method: markdown_header_based\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(chunk_text)
                    logger.info(f"Debug: Saved chunk {chunk_num} to {debug_file}")

                try:
                    # Generate embedding
                    embedding = await self._get_embedding(chunk_text)
                    
                    # Create document object for vector storage
                    doc_id = f"{file_hash}_page_{page_chunk['page_number']}"
                    
                    document = {
                        "docId": doc_id,
                        "content": chunk_text,
                        "content_vector": embedding,
                        "title": f"{file_name} - Page {page_chunk['page_number']}",
                        "file_name": file_name,
                        "file_hash": file_hash,
                        "page_number": page_chunk['page_number'],
                        "paragraph_number": 1,  # Page-level, so always 1
                        "chunk_number": chunk_num,
                        "document_type": document_type,
                        "company": company,
                        "industry": industry,
                        "report_year": report_year,
                        "source": file_path,
                        "metadata": json.dumps({
                            "total_pages": page_count,
                            "total_page_chunks": len(page_chunks),
                            "chunk_tokens": page_chunk['tokens'],
                            "chunk_length": page_chunk['length'],
                            "file_size": Path(file_path).stat().st_size,
                            "processing_method": "page_based_chunking",
                            "overlap_percentage": 10
                        })
                    }
                    
                    documents.append(document)
                    logger.info(f"Processed page chunk {chunk_num} (page {page_chunk['page_number']}): {page_chunk['tokens']} tokens, {page_chunk['length']} characters")
                
                except Exception as e:
                    logger.error(f"Error processing page chunk {chunk_num}: {e}")
                    continue
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in page chunking: {e}")
            return []
    
    async def _process_pdf_file(
        self, 
        file_path: str, 
        original_filename: str,
        document_type: str,
        company: str,
        industry: str,
        report_year: str,
        is_debug: bool = False
    ) -> List[Dict]:
        """Process PDF file using the configured chunking method."""
        try:
            # Process PDF with Document Intelligence
            with open(file_path, "rb") as file:
                poller = self.doc_intelligence_client.begin_analyze_document(
                    # model_id="prebuilt-layout",
                    # body=file, 
                    # output_content_format=DocumentContentFormat.MARKDOWN
                    "prebuilt-layout",
                    AnalyzeDocumentRequest(bytes_source=file.read()),
                    output_content_format=DocumentContentFormat.MARKDOWN,
                )
            
            result: AnalyzeResult = poller.result()

            
            if is_debug:
                # Save result.content to file for debugging
                debug_dir = Path("debug_content")
                debug_dir.mkdir(exist_ok=True)
                content_file = debug_dir / f"{original_filename}_content.txt"
                with open(content_file, "w", encoding="utf-8") as f:
                    f.write(f"=== DOCUMENT INTELLIGENCE CONTENT DEBUG ===\n")
                    f.write(f"File: {original_filename}\n")
                    f.write(f"Content Format: {result.content_format}\n")
                    f.write(f"Content Length: {len(result.content) if result.content else 0}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(result.content if result.content else "No content extracted")
                logger.info(f"Debug: Saved content to {content_file}")

            page_count = len(result.pages) if result.pages else 0
            
            logger.info(f"Document Intelligence processing complete: {page_count} pages")
            
            # Route to appropriate chunking method based on configuration
            if self.processing_method == "semantic":
                documents = await self._process_pdf_with_semantic_chunking(
                    file_path, original_filename, document_type, company, industry, report_year, result, is_debug
                )
            elif self.processing_method == "page":
                documents = await self._process_pdf_with_page_chunking(
                    file_path, original_filename, document_type, company, industry, report_year, result, is_debug
                )
            else:
                # Fallback to semantic
                logger.warning(f"Unknown processing method: {self.processing_method}. Using semantic chunking.")
                documents = await self._process_pdf_with_semantic_chunking(
                    file_path, original_filename, document_type, company, industry, report_year, result, is_debug
                )
            
            logger.info(f"Processed {len(documents)} chunks using {self.processing_method} chunking from {page_count} pages")
            
            # Log token distribution for analysis
            if documents:
                token_counts = [get_token_numbers(doc["content"], "gpt-4") for doc in documents]
                avg_tokens = sum(token_counts) / len(token_counts)
                min_tokens = min(token_counts)
                max_tokens = max(token_counts)
                logger.info(f"Token distribution - Avg: {avg_tokens:.1f}, Min: {min_tokens}, Max: {max_tokens}")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing PDF file {file_path}: {e}")
            return []

    async def _upload_documents_to_vector_db(self, documents: List[Dict]) -> bool:
        """Upload processed documents to Azure AI Search."""
        try:
            if not documents:
                return False
            
            # Upload documents in batches
            batch_size = 50
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                try:
                    result = self.search_client.upload_documents(documents=batch)
                    logger.info(f"Uploaded batch {i//batch_size + 1} with {len(batch)} documents")
                    
                    # Check for any failed uploads
                    for res in result:
                        if not res.succeeded:
                            logger.error(f"Failed to upload document {res.key}: {res.error_message}")
                            
                except Exception as e:
                    logger.error(f"Error uploading batch {i//batch_size + 1}: {e}")
                    return False
            
            logger.info(f"Successfully uploaded {len(documents)} documents to vector database")
            return True
        except Exception as e:
            logger.error(f"Error processing PDF file {e}")
            return []

    @kernel_function(
        description="Handle user file upload requests and provide guidance",
        name="handle_upload_request"
    )
    async def handle_upload_request(self, user_message: str) -> str:
        """
        Detect and handle file upload requests from users.
        
        Args:
            user_message: User's message content
            
        Returns:
            JSON string with upload guidance
        """
        # Check for upload-related keywords
        upload_keywords = [
            "upload", "업로드", "파일 올리기", "문서 올리기", 
            "file upload", "add document", "add file"
        ]
        
        message_lower = user_message.lower()
        wants_upload = any(keyword in message_lower for keyword in upload_keywords)
        
        if wants_upload:
            response = {
                "status": "upload_requested",
                "message": """📁 **File Upload Available**

I can help you upload and process documents for AI analysis!

**Supported Files:** PDF, DOCX, TXT (up to 10 files)
**Process:** Files are chunked by page and stored for semantic search

**To Upload:**
1. Use your application's file upload interface
2. Select up to 10 files
3. Provide metadata (optional): document type, company, industry, year
4. Files will be processed in background
5. You'll get confirmation when ready

**After Upload:**
I can search, analyze, and answer questions about your documents.

Ready to upload files?""",
                "action_required": "file_upload"
            }
        else:
            response = {
                "status": "no_upload_needed",
                "message": "No file upload detected in your message.",
                "action_required": None
            }
        
        return json.dumps(response)
    
    @kernel_function(
        description="Upload and process documents for AI Search. Call this when files need to be processed as unstructured document data.",
        name="upload_documents"  
    )
    async def upload_documents(
        self,
        file_paths: str,  # JSON string of file paths
        file_names: str = "",  # JSON string of original filenames 추가
        document_type: str = "GENERAL",
        company: str = "",
        industry: str = "",
        report_year: str = "",
        force_upload: str = "false",
        is_debug: bool = False
    ) -> str:
        """Upload and process multiple files for document search."""
        try:
            # Parse file paths and names from JSON strings
            try:
                file_paths_list = json.loads(file_paths)
                file_names_list = json.loads(file_names) if file_names else []
            except json.JSONDecodeError:
                # If not JSON, assume single file path
                file_paths_list = [file_paths]
                file_names_list = [Path(file_paths).name]
        
            # Ensure we have matching file names
            if len(file_names_list) != len(file_paths_list):
                file_names_list = [Path(fp).name for fp in file_paths_list]
        
            logger.info(f"Starting upload for {len(file_paths_list)} files")
        
            # Convert string boolean to actual boolean
            force_upload_bool = force_upload.lower() == "true"
        
            results = []
            total_documents = 0
        
            for file_path, original_filename in zip(file_paths_list, file_names_list):
                try:
                    # Check if file exists
                    if not Path(file_path).exists():
                        results.append({
                            "file": original_filename,  # 실제 파일명으로 표시
                            "status": "error",
                            "message": "File not found"
                        })
                        continue
                
                    # Generate file hash for duplicate check
                    file_hash = self._generate_doc_hash(file_path)
                
                    # Check if file already exists in vector DB
                    if not force_upload_bool and await self._file_exists_in_vector_db(file_hash):
                        results.append({
                            "file": original_filename,
                            "status": "skipped",
                            "message": "File already exists in vector database"
                        })
                        continue
                
                    # Process file based on extension
                    file_extension = Path(original_filename).suffix.lower()
                
                    if file_extension == '.pdf':
                        documents = await self._process_pdf_file(
                            file_path, original_filename, document_type, company, industry, report_year, is_debug
                        )
                    else:
                        results.append({
                            "file": original_filename,
                            "status": "error",
                            "message": f"Unsupported file type: {file_extension}"
                        })
                        continue
                
                    if documents:
                        # Upload to vector database
                        if await self._upload_documents_to_vector_db(documents):
                            results.append({
                                "file": original_filename,
                                "status": "success",
                                "message": f"Successfully uploaded {len(documents)} chunks",
                                "chunks_count": len(documents)
                            })
                            total_documents += len(documents)
                        else:
                            results.append({
                                "file": original_filename,
                                "status": "error",
                                "message": "Failed to upload to vector database"
                            })
                        
                    else:
                        results.append({
                            "file": original_filename,
                            "status": "error",
                            "message": "No content could be extracted from file"
                        })
                        
                except Exception as e:
                    logger.error(f"Error processing file {original_filename}: {e}")
                    results.append({
                        "file": original_filename,
                        "status": "error",
                        "message": str(e)
                    })
        
            # Prepare final response
            successful_uploads = [r for r in results if r["status"] == "success"]
        
            response = {
                "status": "completed",
                "total_files": len(file_paths_list),
                "successful_uploads": len(successful_uploads),
                "total_documents_uploaded": total_documents,
                "results": results
            }
        
            logger.info(f"Upload completed: {len(successful_uploads)}/{len(file_paths_list)} files successful")
            return json.dumps(response)
            
        except Exception as e:
            error_msg = f"Error in file upload: {str(e)}"
            logger.error(error_msg)
            return json.dumps({
                "status": "error", 
                "message": error_msg,
                "results": []
            })
    
    
    
    @kernel_function(
        description="Check if documents exist in the vector database",
        name="check_docs_status"
    )
    async def check_docs_status(self, file_names: str) -> str:
        """
        Check the status of documents in the vector database.

        Args:
            file_names: Comma-separated string of file names to check
            
        Returns:
            JSON string with file status information
        """
        try:
            # Parse file names
            files_list = [name.strip() for name in file_names.split(",")]
            
            results = {}
            
            for file_name in files_list:
                try:
                    # Search for documents with this file name
                    search_results = self.search_client.search(
                        search_text="*",
                        filter=f"file_name eq '{file_name}'",
                        top=1
                    )
                    
                    exists = False
                    for _ in search_results:
                        exists = True
                        break
                    
                    results[file_name] = {
                        "exists": exists,
                        "status": "found" if exists else "not_found"
                    }
                    
                except Exception as e:
                    logger.error(f"Error checking file {file_name}: {e}")
                    results[file_name] = {
                        "exists": False,
                        "status": "error",
                        "error": str(e)
                    }
            
            existing_files = [f for f, info in results.items() if info["exists"]]
            missing_files = [f for f, info in results.items() if not info["exists"]]
            
            response = {
                "status": "success",
                "total_checked": len(files_list),
                "existing_files": existing_files,
                "missing_files": missing_files,
                "results": results
            }
            
            return json.dumps(response)
            
        except Exception as e:
            logger.error(f"Error checking file status: {e}")
            return json.dumps({
                "status": "error",
                "message": str(e)
            })
    
    @kernel_function(
        description="Format upload completion notification for users",
        name="notify_upload_completion"
    )
    async def notify_upload_completion(self, upload_results: str) -> str:
        """
        Process and format upload completion notification.
        
        Args:
            upload_results: JSON string containing upload results
            
        Returns:
            JSON string with formatted completion message
        """
        try:
            results = json.loads(upload_results)
            
            total_files = results.get("total_files", 0)
            successful = results.get("successful_uploads", 0)
            failed = total_files - successful
            
            if successful > 0:
                message = f"""🎉 **Upload Complete!**

📊 **Summary:**
• Files Processed: {total_files}
• Successfully Uploaded: {successful}
• Failed: {failed}

✅ **Your documents are ready for AI search!**

**What's next:**
• Ask questions about your documents
• Request analysis or summaries
• Search for specific information
• Compare data across documents

**Example questions:**
• "What are the key findings in my documents?"
• "Summarize the main points from the uploaded reports"
• "Find information about revenue growth"

How can I help analyze your documents?"""
            else:
                message = f"""⚠️ **Upload Issues**

📊 **Summary:**
• Total Files: {total_files}
• Failed: {failed}

**Common issues:**
• Unsupported file format (only PDF, DOCX, TXT)
• File corruption or encryption
• Network issues

**Next steps:**
• Check file formats
• Try smaller files
• Upload one file at a time

Would you like to try again?"""
            
            response = {
                "status": "completed",
                "message": message,
                "successful_uploads": successful,
                "failed_uploads": failed,
                "ready_for_analysis": successful > 0
            }
            
            return json.dumps(response)
            
        except Exception as e:
            logger.error(f"Error formatting completion message: {e}")
            return json.dumps({
                "status": "error",
                "message": "Error processing upload notification."
            })
    
    def _get_chunk_page_number(self, chunk_text: str, result) -> int:
        """
        Document Intelligence의 paragraph 정보를 활용해서 청크의 페이지 번호를 찾음
        """
        if not result.paragraphs or not chunk_text.strip():
            return 1
    
        # 청크에서 검색할 텍스트 추출 (첫 번째 의미있는 라인)
        chunk_lines = [line.strip() for line in chunk_text.split('\n') if line.strip()]
        if not chunk_lines:
            return 1
        
        # 검색 텍스트 준비 (헤더 마크다운 제거)
        search_text = chunk_lines[0]
        clean_search = re.sub(r'^#+\s*', '', search_text).strip()
        
        # 1단계: 정확한 텍스트 매칭
        for paragraph in result.paragraphs:
            if not paragraph.content or not paragraph.bounding_regions:
                continue
                
            # 원본 텍스트 매칭
            if search_text.lower() in paragraph.content.lower():
                return paragraph.bounding_regions[0].page_number
            
            # 헤더 제거 버전 매칭
            if clean_search and clean_search.lower() in paragraph.content.lower():
                return paragraph.bounding_regions[0].page_number
        
        # 2단계: 짧은 버전으로 재시도
        short_text = clean_search[:30] if clean_search else search_text[:30]
        for paragraph in result.paragraphs:
            if paragraph.content and paragraph.bounding_regions:
                if short_text.lower() in paragraph.content.lower():
                    return paragraph.bounding_regions[0].page_number
        
        # 3단계: 첫 번째 단어로 매칭
        words = clean_search.split() if clean_search else search_text.split()
        if words and len(words[0]) > 3:
            first_word = words[0].lower()
            for paragraph in result.paragraphs:
                if paragraph.content and paragraph.bounding_regions:
                    if first_word in paragraph.content.lower():
                        return paragraph.bounding_regions[0].page_number
        
        logger.warning(f"Could not find page number for chunk: {search_text[:50]}...")
        return 1


