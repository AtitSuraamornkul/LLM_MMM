
from langchain.schema import Document
from datetime import datetime
import re

def format_marketing_analysis_for_rag_from_file(file_path):
    """Read marketing analysis from txt file and format for RAG/vector database storage"""
    
    # Read the file
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
    
    timestamp = datetime.now().isoformat()
    documents = []
    
    # Split content into sections based on headers/titles
    sections = []
    
    # Define section patterns to split the content
    section_patterns = [
        r"Channel Contribution Analysis:",
        r"Marketing Channel Spend and ROI Analysis:",
        r"=== MONTHLY CHANNEL PERFORMANCE WITH ANOMALIES ===",
        r"=== QUARTERLY CHANNEL TRENDS ===",
        r"=== SPIKE AND DIP ANALYSIS ===",
        r"=== CHANNEL COMPARISON ===",
        r"=== MOMENTUM ANALYSIS ===",
        r"Title: Return on investment",
        r"ROI vs Effectiveness Analysis:",
        r"ROI vs Marginal ROI Performance Analysis:",
        r"ROI and CPIK Performance Analysis with Confidence Intervals:",
        r"Marketing Response Curves Performance Analysis:"
    ]
    
    # Split content by sections
    current_pos = 0
    for i, pattern in enumerate(section_patterns):
        match = re.search(pattern, content[current_pos:])
        if match:
            start_pos = current_pos + match.start()
            
            # Find the end of this section (start of next section or end of content)
            end_pos = len(content)
            for next_pattern in section_patterns[i+1:]:
                next_match = re.search(next_pattern, content[start_pos:])
                if next_match:
                    end_pos = start_pos + next_match.start()
                    break
            
            section_content = content[start_pos:end_pos].strip()
            sections.append((pattern, section_content))
            current_pos = end_pos
    
    # Document 1: Channel Contribution Analysis
    contrib_content = next((content for pattern, content in sections if "Channel Contribution Analysis" in pattern), "")
    if contrib_content:
        # Extract key metrics
        baseline_pct = re.search(r"Baseline revenue accounts for ([\d.]+)%", contrib_content)
        marketing_pct = re.search(r"Marketing channels drive ([\d.]+)%", contrib_content)
        
        document_1 = Document(
            page_content=contrib_content,
            metadata={
                "source": "marketing_analysis",
                "analysis_type": "channel_contribution",
                "timestamp": timestamp,
                "channels": ["DV_360_X1", "GOOGLE", "META", "TIKTOK"],
                "baseline_revenue_share": float(baseline_pct.group(1)) if baseline_pct else None,
                "marketing_revenue_share": float(marketing_pct.group(1)) if marketing_pct else None,
                "top_channel": "DV_360_X1"
            }
        )
        documents.append(document_1)
    
    # Document 2: Spend and ROI Analysis
    spend_roi_content = next((content for pattern, content in sections if "Marketing Channel Spend and ROI Analysis" in pattern), "")
    if spend_roi_content:
        # Extract average ROI
        avg_roi = re.search(r"Average ROI across all channels: ([\d.]+)x", spend_roi_content)
        
        document_2 = Document(
            page_content=spend_roi_content,
            metadata={
                "source": "marketing_analysis",
                "analysis_type": "spend_roi",
                "timestamp": timestamp,
                "average_roi": float(avg_roi.group(1)) if avg_roi else None,
                "highest_roi_channel": "META",
                "lowest_roi_channel": "DV_360_X1",
                "most_efficient_channel": "META",
                "least_efficient_channel": "DV_360_X1"
            }
        )
        documents.append(document_2)
    
    # Document 3: Monthly Performance Analysis
    monthly_content = next((content for pattern, content in sections if "MONTHLY CHANNEL PERFORMANCE" in pattern), "")
    if monthly_content:
        document_3 = Document(
            page_content=monthly_content,
            metadata={
                "source": "marketing_analysis",
                "analysis_type": "monthly_performance",
                "timestamp": timestamp,
                "most_stable_channel": "META",
                "most_volatile_channel": "TIKTOK",
                "analysis_period": "30_months",
                "performance_type": "anomaly_detection"
            }
        )
        documents.append(document_3)
    
    # Document 4: Quarterly Trends
    quarterly_content = next((content for pattern, content in sections if "QUARTERLY CHANNEL TRENDS" in pattern), "")
    if quarterly_content:
        document_4 = Document(
            page_content=quarterly_content,
            metadata={
                "source": "marketing_analysis",
                "analysis_type": "quarterly_trends",
                "timestamp": timestamp,
                "trend_analysis": True,
                "performance_type": "quarterly_breakdown"
            }
        )
        documents.append(document_4)
    
    # Document 5: Channel Comparison and Rankings
    comparison_content = next((content for pattern, content in sections if "CHANNEL COMPARISON" in pattern), "")
    spike_content = next((content for pattern, content in sections if "SPIKE AND DIP ANALYSIS" in pattern), "")
    momentum_content = next((content for pattern, content in sections if "MOMENTUM ANALYSIS" in pattern), "")
    
    combined_comparison = f"{comparison_content}\n\n{spike_content}\n\n{momentum_content}".strip()
    if combined_comparison:
        document_5 = Document(
            page_content=combined_comparison,
            metadata={
                "source": "marketing_analysis",
                "analysis_type": "channel_comparison_rankings",
                "timestamp": timestamp,
                "includes_momentum": True,
                "includes_volatility": True,
                "ranking_type": "comprehensive"
            }
        )
        documents.append(document_5)
    
    # Document 6: ROI Insights
    roi_insights_content = next((content for pattern, content in sections if "Title: Return on investment" in pattern), "")
    if roi_insights_content:
        document_6 = Document(
            page_content=roi_insights_content,
            metadata={
                "source": "marketing_analysis",
                "analysis_type": "roi_insights",
                "timestamp": timestamp,
                "highest_roi_channel": "META",
                "highest_effectiveness_channel": "DV_360_X1",
                "lowest_cpik_channel": "META"
            }
        )
        documents.append(document_6)
    
    # Document 7: ROI vs Effectiveness Analysis
    effectiveness_content = next((content for pattern, content in sections if "ROI vs Effectiveness Analysis" in pattern), "")
    if effectiveness_content:
        document_7 = Document(
            page_content=effectiveness_content,
            metadata={
                "source": "marketing_analysis",
                "analysis_type": "roi_effectiveness",
                "timestamp": timestamp,
                "total_spend": 2340830,
                "analysis_focus": "efficiency_optimization",
                "performance_categories": ["High Potential", "Optimization Needed", "Cost Efficient"]
            }
        )
        documents.append(document_7)
    
    # Document 8: ROI vs Marginal ROI Analysis
    marginal_roi_content = next((content for pattern, content in sections if "ROI vs Marginal ROI Performance Analysis" in pattern), "")
    if marginal_roi_content:
        document_8 = Document(
            page_content=marginal_roi_content,
            metadata={
                "source": "marketing_analysis",
                "analysis_type": "marginal_roi",
                "timestamp": timestamp,
                "average_marginal_roi": 2.0,
                "saturation_analysis": True,
                "diminishing_returns_assessment": True
            }
        )
        documents.append(document_8)
    
    # Document 9: ROI and CPIK with Confidence Intervals
    confidence_content = next((content for pattern, content in sections if "ROI and CPIK Performance Analysis with Confidence Intervals" in pattern), "")
    if confidence_content:
        document_9 = Document(
            page_content=confidence_content,
            metadata={
                "source": "marketing_analysis",
                "analysis_type": "roi_cpik_confidence",
                "timestamp": timestamp,
                "confidence_level": "90_percent",
                "most_reliable_channel": "TIKTOK",
                "least_reliable_channel": "DV_360_X1",
                "statistical_significance": True,
                "uncertainty_analysis": True
            }
        )
        documents.append(document_9)
    
    # Document 10: Response Curves Analysis
    response_curves_content = next((content for pattern, content in sections if "Marketing Response Curves Performance Analysis" in pattern), "")
    if response_curves_content:
        document_10 = Document(
            page_content=response_curves_content,
            metadata={
                "source": "marketing_analysis",
                "analysis_type": "response_curves",
                "timestamp": timestamp,
                "total_spend": 2340830,
                "total_revenue": 10978873,
                "overall_roi": 4.69,
                "scenario_analysis": True,
                "marginal_returns": True,
                "growth_potential": True,
                "spend_scenarios": ["25%", "50%", "100%"]
            }
        )
        documents.append(document_10)
    
    return documents


def main():
    """Main function to process marketing analysis file"""
    
    # Specify your file path
    file_path = "summary_extract_output.txt"  # Change this to your file path
    
    print("Reading and processing marketing analysis file...")
    documents = format_marketing_analysis_for_rag_from_file(file_path)
    
    print(f"Created {len(documents)} documents for RAG:")
    for i, doc in enumerate(documents, 1):
        print(f"{i}. {doc.metadata['analysis_type']} - {len(doc.page_content)} characters")
        print(f"   Metadata: {doc.metadata}")
        print()
    
    # Optionally save to Pinecone (uncomment and configure)
    # print("Saving to Pinecone...")
    # vectorstore = save_documents_to_pinecone(documents)
    # print("Documents saved to Pinecone successfully!")
    
    # Save documents to JSON for inspection
    import json
    output_data = []
    for doc in documents:
        output_data.append({
            "page_content": doc.page_content,
            "metadata": doc.metadata
        })
    
    with open("marketing_analysis_rag_documents.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print("Documents also saved to 'marketing_analysis_rag_documents.json' for inspection")
    
    return documents

if __name__ == "__main__":
    documents = main()