import json
import logging

logger = logging.getLogger(__name__)

def clean_and_validate_json(content: str) -> str:
        """JSON 응답을 정리하고 검증"""
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
            
            # JSON 검증
            parsed = json.loads(content)
            
            # 재직렬화하여 형식 정리
            clean_json = json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
            
            logger.info(f"[GroupChat] Successfully cleaned and validated JSON")
            return clean_json
            
        except json.JSONDecodeError as e:
            logger.error(f"[GroupChat] JSON validation failed: {e}")
            logger.error(f"[GroupChat] Problematic content: {content[:500]}...")
            
            # 최소한의 fallback JSON 생성
            return json.dumps({
                "sub_topic": "Unknown",
                "final_answer": content[:1000] if content else "No response generated",
                "error": "json_parsing_failed"
            }, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"[GroupChat] Unexpected error in JSON cleaning: {e}")
            return json.dumps({
                "sub_topic": "Unknown", 
                "final_answer": "Processing error occurred",
                "error": str(e)
            }, ensure_ascii=False)

def clean_duplicate_table_content(markdown: str) -> str:
    """Remove text that duplicates table row content."""
    import re
    
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