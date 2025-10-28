import json
import logging
import re

logger = logging.getLogger(__name__)

def _clean_markdown_content(markdown: str) -> str:
    """
    Internal helper to clean escaped characters in markdown content.
    Handles escaped newlines (\\n) and other escape sequences.
    """
    if not markdown or not isinstance(markdown, str):
        return markdown
    
    # Replace escaped newlines with actual newlines
    markdown = markdown.replace('\\n', '\n')
    markdown = markdown.replace('\\t', '\t')
    markdown = markdown.replace('\\r', '\r')
    
    return markdown

def clean_and_validate_json(content: str, return_dict: bool = False) -> str | dict:
        """
        JSON 응답을 정리하고 검증하며, 마크다운 필드의 이스케이프 문자를 자동으로 정리
        
        Args:
            content: JSON 문자열
            return_dict: True면 dict 반환, False면 JSON 문자열 반환 (기본값)
        
        Returns:
            정리된 JSON 문자열 또는 dict 객체
        """
        try:
            # 앞뒤 공백 제거
            content = content.strip()
            
            # markdown 코드 블록이나 설명 텍스트 제거
            if content.startswith('```'):
                # ```json으로 시작하는 경우
                lines = content.split('\n')
                json_lines = []
                in_json = False
                for line in lines:
                    if line.strip().startswith('```'):
                        if not in_json:
                            in_json = True
                        else:
                            break
                    elif in_json:
                        json_lines.append(line)
                content = '\n'.join(json_lines).strip()
            
            # JSON 부분만 추출 (첫 번째 { 부터 마지막 } 까지)
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                content = content[start_idx:end_idx + 1]
            
            # ✅ JSON 파싱 전에 일반적인 문제 수정
            # 1. trailing comma 제거
            content = re.sub(r',(\s*[}\]])', r'\1', content)
            
            # JSON 검증
            parsed = json.loads(content)
            
            # ✅ 파싱 성공 후 마크다운 필드들의 이스케이프 문자 정리
            markdown_fields = [
                'draft_answer_markdown',
                'revised_answer_markdown', 
                'final_answer_markdown',
                'answer_markdown',
                'final_answer',
                'answer'
            ]
            
            for field in markdown_fields:
                if field in parsed and isinstance(parsed[field], str):
                    # 1. 이스케이프 문자 정리
                    parsed[field] = _clean_markdown_content(parsed[field])
                    # 2. 테이블 간격 보정
                    parsed[field] = ensure_table_spacing(parsed[field])
                    # 3. 테이블 중복 본문 제거
                    parsed[field] = clean_duplicate_table_content(parsed[field])
            
            logger.info(f"[GroupChat] Successfully cleaned and validated JSON")
            
            # ✅ return_dict가 True면 dict 반환, 아니면 JSON 문자열 반환
            if return_dict:
                return parsed
            else:
                # 재직렬화하여 형식 정리 (이때 다시 이스케이프됨 주의)
                clean_json = json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
                return clean_json
            
        except json.JSONDecodeError as e:
            logger.error(f"[GroupChat] JSON validation failed: {e}")
            logger.error(f"[GroupChat] Problematic content: {content[:500]}...")
            
            # ✅ 더 aggressive한 복구 시도
            try:
                # 1. 마지막 완전한 } 찾기 (중첩된 객체 고려)
                brace_count = 0
                last_valid_pos = -1
                for i, char in enumerate(content):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            last_valid_pos = i + 1
                            break
                
                if last_valid_pos > 0:
                    content = content[:last_valid_pos]
                    parsed = json.loads(content)
                    
                    # 마크다운 필드 정리
                    for field in markdown_fields:
                        if field in parsed and isinstance(parsed[field], str):
                            # 1. 이스케이프 문자 정리
                            parsed[field] = _clean_markdown_content(parsed[field])
                            # 2. 테이블 간격 보정
                            parsed[field] = ensure_table_spacing(parsed[field])
                            # 3. 테이블 중복 본문 제거
                            parsed[field] = clean_duplicate_table_content(parsed[field])
                    
                    logger.info(f"[GroupChat] Recovered JSON after truncation")
                    
                    if return_dict:
                        return parsed
                    else:
                        clean_json = json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
                        return clean_json
                    
            except Exception as recovery_error:
                logger.error(f"[GroupChat] Recovery attempt failed: {recovery_error}")
            
            # 최소한의 fallback
            fallback = {
                "sub_topic": "Unknown",
                "final_answer": content[:1000] if content else "No response generated",
                "error": "json_parsing_failed"
            }
            
            if return_dict:
                return fallback
            else:
                return json.dumps(fallback, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"[GroupChat] Unexpected error in JSON cleaning: {e}")
            fallback = {
                "sub_topic": "Unknown", 
                "final_answer": "Processing error occurred",
                "error": str(e)
            }
            
            if return_dict:
                return fallback
            else:
                return json.dumps(fallback, ensure_ascii=False)

def clean_duplicate_table_content(markdown: str) -> str:
    """Remove text that duplicates table row content."""
    
    # Find all markdown tables
    table_pattern = r'\|[^\n]+\|[\n\r]+\|[-:\s|]+\|[\n\r]+((?:\|[^\n]+\|[\n\r]+)+)'
    tables = re.finditer(table_pattern, markdown)
    
    cleaned = markdown
    for table_match in tables:
        table_content = table_match.group(0)
        # Extract cell content from table
        cell_pattern = r'\|\s*([^|]+?)\s*\|'
        cells = re.findall(cell_pattern, table_content)
        
        # Remove sentences that contain exact cell content before the table
        for cell in cells:
            cell = cell.strip()
            if len(cell) > 10:  # Only check substantial content
                # Remove lines that contain this cell content before the table
                pattern = f"[^\n]*{re.escape(cell)}[^\n]*\n"
                cleaned = re.sub(pattern, "", cleaned, count=1)
    
    return cleaned

def clean_markdown_escapes(markdown: str) -> str:
    """
    Clean escaped characters in markdown content.
    Handles escaped newlines (\\n) and other escape sequences.
    
    This is a public wrapper around _clean_markdown_content for direct use.
    
    Args:
        markdown: Markdown text that may contain escaped characters
        
    Returns:
        Cleaned markdown with proper formatting
    """
    return _clean_markdown_content(markdown)

def ensure_table_spacing(markdown: str) -> str:
    """
    Ensure markdown tables have blank lines before and after them.
    This fixes common LLM mistakes where tables are placed directly after text.
    
    Args:
        markdown: Markdown text that may contain improperly spaced tables
        
    Returns:
        Markdown with properly spaced tables
    """
    if not markdown:
        return markdown
    
    import re
    
    # Pattern to match markdown tables
    # Matches: | header | ... | followed by separator row and data rows
    table_pattern = r'(\|[^\n]+\|(?:\n\|[-:\s|]+\|)?(?:\n\|[^\n]+\|)*)'
    
    def add_spacing(match):
        table = match.group(0)
        # Check if there's content before the table (not just whitespace/newline)
        start_pos = match.start()
        # Look back to see if there's text immediately before
        if start_pos > 0:
            before_table = markdown[max(0, start_pos - 2):start_pos]
            # If not already separated by blank line, add one
            if before_table and not before_table.endswith('\n\n'):
                table = '\n' + table
        
        # Check if there's content after the table
        end_pos = match.end()
        if end_pos < len(markdown):
            after_table = markdown[end_pos:min(len(markdown), end_pos + 2)]
            # If not already separated by blank line, add one
            if after_table and not after_table.startswith('\n\n'):
                table = table + '\n'
        
        return table
    
    # Apply spacing fix to all tables
    result = re.sub(table_pattern, add_spacing, markdown, flags=re.MULTILINE)
    
    return result
