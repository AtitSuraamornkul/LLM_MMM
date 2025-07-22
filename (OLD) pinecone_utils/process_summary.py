import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import json
from datetime import datetime

@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any]

class MMMSummaryReportProcessor:
    def __init__(self):
        self.section_patterns = {
            'channel_contribution': r'Channel Contribution Analysis:',
            'revenue_attribution': r'Revenue Attribution:',
            'marketing_channel_performance': r'Marketing Channel Performance:',
            'key_insights': r'Key Insights:',
            'methodology': r'Methodology:',
            'spend_roi_analysis': r'Marketing Channel Spend and ROI Analysis:',
            'channel_performance_roi': r'Channel Performance by ROI (Within Media Channel Only):',
            'channel_efficiency': r'Channel Efficiency Analysis:',
            'roi_performance': r'ROI Performance:',
            'budget_allocation_insights': r'Budget Allocation Insights:',
            'monthly_performance': r'=== MONTHLY CHANNEL PERFORMANCE WITH ANOMALIES ===',
            'quarterly_trends': r'=== QUARTERLY CHANNEL TRENDS ===',
            'spike_dip_analysis': r'=== SPIKE AND DIP ANALYSIS ===',
            'channel_comparison': r'=== CHANNEL COMPARISON ===',
            'momentum_analysis': r'=== MOMENTUM ANALYSIS ===',
            'roi_insights': r'Title: Return on investment',
            'roi_effectiveness': r'ROI vs Effectiveness Analysis:',
            'roi_marginal': r'ROI vs Marginal ROI Performance Analysis:',
            'roi_cpik': r'ROI and CPIK Performance Analysis with Confidence Intervals:',
            'response_curves': r'Marketing Response Curves Performance Analysis:'
        }
    
    def find_all_section_boundaries(self, text: str) -> List[Tuple[int, str, str]]:
        """Find all section boundaries in the text"""
        boundaries = []
        
        # Find major section headers
        for section_key, pattern in self.section_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                boundaries.append((match.start(), section_key, pattern.replace(':', '').strip()))
        
        # Find individual channel sections in monthly/quarterly analysis
        channel_section_patterns = [
            (r'(\w+) - Monthly Performance:', 'monthly_channel_detail'),
            (r'(\w+) - Quarterly Analysis:', 'quarterly_channel_detail'),
            (r'(\w+) - 6-Month Momentum Analysis:', 'momentum_channel_detail')
        ]
        
        for pattern, section_type in channel_section_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                channel_name = match.group(1)
                boundaries.append((match.start(), section_type, f'{section_type}: {channel_name}'))
        
        # Find performance ranking sections
        ranking_patterns = [
            (r'Total Revenue Rankings:', 'revenue_rankings'),
            (r'Consistency Rankings:', 'consistency_rankings'),
            (r'Peak Performance Rankings:', 'peak_rankings'),
            (r'ROI Rankings:', 'roi_rankings'),
            (r'Effectiveness Rankings:', 'effectiveness_rankings'),
            (r'Marginal ROI Rankings:', 'marginal_roi_rankings'),
            (r'CPIK Performance with 90% Credible Intervals:', 'cpik_performance'),
            (r'ROI Performance with 90% Credible Intervals:', 'roi_confidence')
        ]
        
        for pattern, section_type in ranking_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                boundaries.append((match.start(), section_type, pattern.replace(':', '').strip()))
        
        # Find analysis subsections
        analysis_patterns = [
            (r'Channel Performance Categories:', 'performance_categories'),
            (r'Strategic Insights:', 'strategic_insights'),
            (r'Saturation Indicators:', 'saturation_indicators'),
            (r'Diminishing Returns Assessment:', 'diminishing_returns'),
            (r'Channel Lifecycle Analysis:', 'lifecycle_analysis'),
            (r'Top Revenue Spikes:', 'revenue_spikes'),
            (r'Seasonal Patterns:', 'seasonal_patterns'),
            (r'Channel Volatility Analysis:', 'volatility_analysis'),
            (r'Spend Allocation:', 'spend_allocation'),
            (r'Combined ROI and CPIK Insights:', 'combined_insights'),
            (r'Current Channel Performance: (TOP 7)', 'current_performance'),
            (r'Spend Increase Scenario Analysis:', 'scenario_analysis'),
            (r'Portfolio Summary:', 'portfolio_summary')
        ]
        
        for pattern, section_type in analysis_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                boundaries.append((match.start(), section_type, pattern.replace(':', '').strip()))
        
        # Sort by position
        boundaries.sort(key=lambda x: x[0])
        return boundaries
    
    def extract_sections_comprehensively(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract all sections ensuring no content is lost"""
        boundaries = self.find_all_section_boundaries(text)
        sections = []
        
        # Handle the case where there might be content before the first boundary
        if boundaries and boundaries[0][0] > 0:
            intro_content = text[:boundaries[0][0]].strip()
            if intro_content:
                sections.append((intro_content, 'introduction', 'Introduction'))
        
        for i, (start_pos, section_type, section_name) in enumerate(boundaries):
            # Determine end position
            if i + 1 < len(boundaries):
                end_pos = boundaries[i + 1][0]
            else:
                end_pos = len(text)
            
            # Extract content
            content = text[start_pos:end_pos].strip()
            if content:
                sections.append((content, section_type, section_name))
        
        return sections
    
    def extract_channel_name_from_content(self, content: str) -> str:
        """Extract channel name from content"""
        # Try to extract from patterns like "CHANNEL_NAME - Analysis:"
        match = re.search(r'(\w+)\s*-\s*(Monthly|Quarterly|Momentum)', content)
        if match:
            return match.group(1).lower()
        
        return "unknown"
    
    def determine_content_type(self, content: str, section_type: str) -> str:
        """Determine the specific content type for better categorization"""
        content_lower = content.lower()
        
        if 'roi' in content_lower and 'performance' in content_lower:
            return 'roi_analysis'
        elif 'revenue' in content_lower and 'attribution' in content_lower:
            return 'revenue_analysis'
        elif 'channel' in content_lower and 'performance' in content_lower:
            return 'channel_performance'
        elif 'budget' in content_lower or 'spend' in content_lower:
            return 'budget_analysis'
        elif 'monthly' in content_lower:
            return 'temporal_analysis'
        elif 'quarterly' in content_lower:
            return 'temporal_analysis'
        elif 'ranking' in content_lower or 'rankings' in content_lower:
            return 'ranking_analysis'
        elif 'insight' in content_lower or 'methodology' in content_lower:
            return 'insights_methodology'
        elif 'confidence' in content_lower or 'interval' in content_lower:
            return 'statistical_analysis'
        elif 'scenario' in content_lower or 'increase' in content_lower:
            return 'scenario_planning'
        else:
            return 'general_analysis'
    
    def process_summary_report(self, file_path: str) -> List[Document]:
        """Process the entire MMM summary report ensuring ALL content is captured"""
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        documents = []
        
        # Get all sections
        sections = self.extract_sections_comprehensively(text)
        
        for content, section_type, section_name in sections:
            if not content.strip():
                continue
            
            # Determine content type
            content_type = self.determine_content_type(content, section_type)
            
            # Base metadata
            metadata = {
                "source": "mmm_summary_report",
                "section": section_type,
                "section_name": section_name,
                "content_type": content_type,
                "content_length": len(content),
                "word_count": len(content.split()),
                "created_at": datetime.now().isoformat()
            }
            
            # Add specific metadata based on section type
            if 'channel' in section_type and section_type.endswith('_detail'):
                channel_name = self.extract_channel_name_from_content(content)
                metadata["channel_name"] = channel_name
                metadata["analysis_type"] = section_type.replace('_channel_detail', '')
                
            elif section_type in ['roi_rankings', 'effectiveness_rankings', 'marginal_roi_rankings']:
                metadata["analysis_focus"] = "performance_ranking"
                
            elif section_type in ['revenue_rankings', 'consistency_rankings', 'peak_rankings']:
                metadata["analysis_focus"] = "channel_ranking"
                
            elif 'confidence' in section_type or 'cpik' in section_type:
                metadata["analysis_focus"] = "statistical_confidence"
                
            elif 'scenario' in section_type:
                metadata["analysis_focus"] = "scenario_planning"
                
            elif section_type in ['channel_contribution', 'revenue_attribution']:
                metadata["analysis_focus"] = "contribution_analysis"
            
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

def process_mmm_summary_report_complete(file_path: str) -> Tuple[List[Document], Dict[str, Any]]:
    """
    Process MMM summary report ensuring ALL content is captured
    
    Returns:
        Tuple of (documents, validation_info)
    """
    processor = MMMSummaryReportProcessor()
    
    # Read original text for validation
    with open(file_path, 'r', encoding='utf-8') as file:
        original_text = file.read()
    
    # Process documents
    documents = processor.process_summary_report(file_path)
    
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
    file_path = "summary_output/summary_extract_output.txt"
    
    # Process summary report
    print("=== Processing MMM Summary Report ===")
    documents, validation = process_mmm_summary_report_complete(file_path)
    save_documents_as_json(documents, "summary_mmm_documents.json")
    
    print(f"Processed {len(documents)} documents")
    print(f"Coverage ratio: {validation['coverage_ratio']:.2%}")
    print(f"Original length: {validation['original_length']} chars")
    print(f"Captured length: {validation['captured_length']} chars")
    print(f"Sections found: {validation['sections_found']}")
    
    # Show sample documents by type
    print(f"\nSample Documents by Type:")
    
    # Group documents by section
    sections = {}
    for doc in documents:
        section = doc.metadata.get('section', 'unknown')
        if section not in sections:
            sections[section] = []
        sections[section].append(doc)
    
    for section, docs in list(sections.items())[:5]:  # Show first 5 sections
        print(f"\n{section.upper()} ({len(docs)} documents):")
        if docs:
            sample_doc = docs[0]
            print(f"  Content preview: {sample_doc.page_content[:150]}...")
            print(f"  Metadata keys: {list(sample_doc.metadata.keys())}")