import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import json
from datetime import datetime

@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any]

class ComprehensiveMMMProcessor:
    def __init__(self):
        self.section_markers = [
            'Time period:',
            'Budget Allocation Table:',
            'Insights:',
            'Chart Title:',
            'Channel:',
            'Percentage Allocation:',
            'Non-optimized Revenue:',
            'Channel revenue change:',
            'Total Optimization Impact'
        ]
    
    def extract_time_period(self, text: str) -> str:
        """Extract time period from the report"""
        match = re.search(r'Time period:\s*(.+)', text)
        return match.group(1).strip() if match else "Unknown"
    
    def find_all_section_boundaries(self, text: str) -> List[Tuple[int, str, str]]:
        """Find all section boundaries in the text"""
        boundaries = []
        
        # Find Time period and summary metrics
        time_match = re.search(r'Time period:', text)
        if time_match:
            boundaries.append((time_match.start(), 'summary_metrics', 'Time period'))
        
        # Find Budget Allocation Table
        budget_match = re.search(r'Budget Allocation Table:', text)
        if budget_match:
            boundaries.append((budget_match.start(), 'budget_allocation', 'Budget Allocation Table'))
        
        # Find Insights
        insights_match = re.search(r'Insights:', text)
        if insights_match:
            boundaries.append((insights_match.start(), 'insights', 'Insights'))
        
        # Find all Chart Titles
        chart_matches = re.finditer(r'Chart Title:\s*(.+)', text)
        for i, match in enumerate(chart_matches):
            chart_title = match.group(1).strip()
            boundaries.append((match.start(), f'chart_{i+1}', f'Chart: {chart_title}'))
        
        # Find Percentage Allocation
        percentage_match = re.search(r'Percentage Allocation:', text)
        if percentage_match:
            boundaries.append((percentage_match.start(), 'percentage_allocation', 'Percentage Allocation'))
        
        # Find Non-optimized Revenue section
        revenue_match = re.search(r'Non-optimized Revenue:', text)
        if revenue_match:
            boundaries.append((revenue_match.start(), 'revenue_details', 'Revenue Details'))
        
        # Find all Channel sections
        channel_matches = re.finditer(r'Channel:\s*(.+)', text)
        for match in channel_matches:
            channel_name = match.group(1).strip().split('\n')[0]
            boundaries.append((match.start(), 'channel_analysis', f'Channel: {channel_name}'))
        
        # Sort by position
        boundaries.sort(key=lambda x: x[0])
        return boundaries
    
    def extract_sections_comprehensively(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract all sections ensuring no content is lost"""
        boundaries = self.find_all_section_boundaries(text)
        sections = []
        
        for i, (start_pos, section_type, section_name) in enumerate(boundaries):
            # Determine end position
            if i + 1 < len(boundaries):
                end_pos = boundaries[i + 1][0]
            else:
                end_pos = len(text)
            
            # Extract content
            content = text[start_pos:end_pos].strip()
            sections.append((content, section_type, section_name))
        
        return sections
    
    def extract_channel_name(self, content: str) -> str:
        """Extract channel name from content"""
        match = re.search(r'Channel:\s*(.+)', content)
        if match:
            return match.group(1).split('\n')[0].strip().lower()
        return "unknown"
    
    def extract_chart_title(self, content: str) -> str:
        """Extract chart title from content"""
        match = re.search(r'Chart Title:\s*(.+)', content)
        if match:
            return match.group(1).strip()
        return "Unknown Chart"
    
    def process_report_comprehensive(self, file_path: str) -> List[Document]:
        """Process the entire MMM report ensuring ALL content is captured"""
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        documents = []
        time_period = self.extract_time_period(text)
        
        # Get all sections
        sections = self.extract_sections_comprehensively(text)
        
        for content, section_type, section_name in sections:
            if not content.strip():
                continue
                
            # Base metadata
            metadata = {
                "source": "mmm_report",
                "section": section_type,
                "section_name": section_name,
                "time_period": time_period,
                "content_length": len(content)
            }
            
            # Add specific metadata based on section type
            if section_type == 'summary_metrics':
                metadata["report_type"] = "optimization_results"
                
            elif section_type == 'budget_allocation':
                metadata["report_type"] = "channel_allocation"
                
            elif section_type == 'insights':
                metadata["report_type"] = "analysis_insights"
                
            elif section_type.startswith('chart_'):
                metadata["report_type"] = "data_visualization"
                metadata["chart_title"] = self.extract_chart_title(content)
                
            elif section_type == 'channel_analysis':
                metadata["report_type"] = "channel_performance"
                metadata["channel_name"] = self.extract_channel_name(content)
                
            elif section_type == 'percentage_allocation':
                metadata["report_type"] = "allocation_breakdown"
                
            elif section_type == 'revenue_details':
                metadata["report_type"] = "revenue_analysis"
            
            documents.append(Document(
                page_content=content,
                metadata=metadata
            ))
        
        return documents
    
    def validate_content_coverage(self, original_text: str, documents: List[Document]) -> Dict[str, Any]:
        """Validate that all content has been captured"""
        total_original_chars = len(original_text.replace('\n', '').replace(' ', ''))
        total_captured_chars = sum(len(doc.page_content.replace('\n', '').replace(' ', '')) for doc in documents)
        
        # Account for some overlap in boundaries
        coverage_ratio = min(total_captured_chars / total_original_chars, 1.0)
        
        return {
            "original_length": len(original_text),
            "captured_length": sum(len(doc.page_content) for doc in documents),
            "coverage_ratio": coverage_ratio,
            "num_documents": len(documents),
            "sections_found": [doc.metadata["section"] for doc in documents]
        }

def process_mmm_report_complete(file_path: str) -> Tuple[List[Document], Dict[str, Any]]:
    """
    Process MMM report ensuring ALL content is captured
    
    Returns:
        Tuple of (documents, validation_info)
    """
    processor = ComprehensiveMMMProcessor()
    
    # Read original text for validation
    with open(file_path, 'r', encoding='utf-8') as file:
        original_text = file.read()
    
    # Process documents
    documents = processor.process_report_comprehensive(file_path)
    
    # Validate coverage
    validation_info = processor.validate_content_coverage(original_text, documents)
    
    return documents, validation_info



def save_documents_as_json(documents: List[Document], output_file: str):
    """Save documents as JSON format"""
    docs_data = []
    
    for i, doc in enumerate(documents):
        doc_dict = {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        docs_data.append(doc_dict)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(docs_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(docs_data)} documents to {output_file}")

# Example usage
if __name__ == "__main__":
    file_path = "llm_input/llm_input.txt"
    
    # Method 1: Comprehensive semantic processing
    print("=== Comprehensive Semantic Processing ===")
    documents, validation = process_mmm_report_complete(file_path)
    save_documents_as_json(documents, "mmm_documents.json")
    
    print(f"Processed {len(documents)} documents")
    print(f"Coverage ratio: {validation['coverage_ratio']:.2%}")
    print(f"Original length: {validation['original_length']} chars")
    print(f"Captured length: {validation['captured_length']} chars")
    print(f"Sections found: {validation['sections_found']}")
    
    