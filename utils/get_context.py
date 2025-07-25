import streamlit as st
import utils.check_token as check_token
from utils.file_processing import analyze_csv_files_dynamically
import re
from collections import defaultdict
from datetime import datetime
import os

MAX_TOKENS = 10000
RESERVED_FOR_ANSWER = 2000 

# extract date / channels

def extract_channels_and_periods_from_text(text_content):
    """Extract channels and time periods from MMM text file"""
    channels = extract_channels_from_text(text_content)
    time_periods = extract_time_periods_from_text(text_content)
    return channels, time_periods

def extract_channels_from_text(text_content):
    """Extract channel names from the text content"""
    channels = set()
    
    # Method 1: From Budget Allocation Table
    budget_allocation_pattern = r'Budget Allocation Table:(.*?)(?=\n\n|\nChart|\nInsights|$)'
    budget_match = re.search(budget_allocation_pattern, text_content, re.DOTALL)
    
    if budget_match:
        budget_section = budget_match.group(1)
        channel_matches = re.findall(r'-\s*([a-zA-Z_]+):\s*\d+%', budget_section)
        channels.update([ch.lower().strip() for ch in channel_matches])
    
    # Method 2: From spend change data
    spend_change_pattern = r'(\w+)\s+Spend Change:'
    spend_matches = re.findall(spend_change_pattern, text_content)
    channels.update([ch.lower().strip() for ch in spend_matches])
    
    # Method 3: From percentage allocation
    percentage_pattern = r'^([a-zA-Z_]+)\s+\d+\.\d+%'
    percentage_matches = re.findall(percentage_pattern, text_content, re.MULTILINE)
    channels.update([ch.lower().strip() for ch in percentage_matches])
    
    # Method 4: From channel-specific sections
    channel_section_pattern = r'Channel:\s*([a-zA-Z_]+)'
    section_matches = re.findall(channel_section_pattern, text_content, re.IGNORECASE)
    channels.update([ch.lower().strip() for ch in section_matches])
    
    # Method 5: From revenue change data
    revenue_pattern = r'^([a-zA-Z_]+)\s+THB[0-9,.-]+\s*\((Increase|Decrease)\)'
    revenue_matches = re.findall(revenue_pattern, text_content, re.MULTILINE)
    channels.update([ch.lower().strip() for ch, _ in revenue_matches])
    
    # Clean up and filter
    channels = list(channels)
    channels = [ch for ch in channels if len(ch) > 1 and ch not in ['thb', 'change', 'increase', 'decrease']]
    
    return sorted(channels)

def extract_time_periods_from_text(text_content):
    """Extract time periods from the text content"""
    time_periods = set()
    
    # Method 1: Extract main time period from header
    main_period_pattern = r'Time period:\s*([^-\n]+)\s*-\s*([^-\n]+)'
    main_match = re.search(main_period_pattern, text_content)
    
    if main_match:
        start_date = main_match.group(1).strip()
        end_date = main_match.group(2).strip()
        years = extract_years_from_dates(start_date, end_date)
        time_periods.update(years)
        time_periods.update(['monthly', 'quarterly'])
    
    # Method 2: Extract specific years
    year_pattern = r'\b(20\d{2})\b'
    year_matches = re.findall(year_pattern, text_content)
    time_periods.update(year_matches)
    
    # Method 3: Extract from historical period mentions
    historical_pattern = r'(\d{4}-\d{2}-\d{2})'
    historical_matches = re.findall(historical_pattern, text_content)
    for match in historical_matches:
        year = match[:4]
        time_periods.add(year)
    
    # Add common time descriptors
    time_periods.update(['trend', 'optimization_period'])
    
    return sorted(list(time_periods))

def extract_years_from_dates(start_date, end_date):
    """Extract years from date strings"""
    years = set()
    
    date_formats = [
        '%b %d, %Y',    # Jul 4, 2022
        '%Y-%m-%d',     # 2022-07-04
        '%m/%d/%Y',     # 07/04/2022
    ]
    
    for date_str in [start_date, end_date]:
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                years.add(str(parsed_date.year))
                break
            except ValueError:
                continue
    
    return years

def load_config_from_file():
    """Load configuration from fixed file path"""
    file_path = "llm_input/llm_input.txt"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text_content = file.read()
        
        channels, time_periods = extract_channels_and_periods_from_text(text_content)
        
        # Only store if we actually found something
        if channels and time_periods:
            st.session_state['company_channels'] = channels
            st.session_state['company_time_periods'] = time_periods
            st.session_state['config_loaded'] = True
            return channels, time_periods
        else:
            # Extraction failed
            st.session_state['config_loaded'] = False
            return None, None
        
    except FileNotFoundError:
        st.warning(f"⚠️ Configuration file not found: {file_path}")
        st.session_state['config_loaded'] = False
        return None, None
    except Exception as e:
        st.error(f"❌ Error reading configuration file: {str(e)}")
        st.session_state['config_loaded'] = False
        return None, None

# config

def get_company_config():
    """Get company-specific configuration with auto-loading from file"""
    
    # Try to load from file if not already loaded
    if not st.session_state.get('config_loaded', False):
        channels, time_periods = load_config_from_file()
        if channels and time_periods:
            st.session_state['company_channels'] = channels
            st.session_state['company_time_periods'] = time_periods
            st.session_state['config_loaded'] = True
    
    return {
        "channels": st.session_state.get('company_channels', []),
        "time_periods": st.session_state.get('company_time_periods', []),
        "currency": "THB",
        "date_format": "YYYY-MM-DD_to_YYYY-MM-DD"
    }

def show_startup_config():
    """Show current configuration on app startup"""
    config = get_company_config()
    
    # Check if we have valid configuration
    if not config["channels"] or not config["time_periods"]:
        st.error("""
        ❌ **Configuration Not Found**
        
        Could not extract channels or time periods from `llm_input/llm_input.txt`
        
        Please check that the file exists and contains valid MMM data.
        """)
        return
    
    # Show successful configuration
    if st.session_state.get('config_loaded', False):
        status_emoji = "✅"
        status_text = "Auto-detected from llm_input.txt"
    else:
        status_emoji = "⚠️"
        status_text = "Configuration issue"
    
    # Format channels and time periods
    channels_text = ", ".join(config["channels"])
    
    # Extract years for time display
    time_periods = config["time_periods"]
    years = sorted([p for p in time_periods if p.isdigit()])
    
    if len(years) >= 2:
        time_display = f"June {years[0]} to July {years[-1]}"
    else:
        time_display = ", ".join(time_periods)
    
    # Show configuration
    st.info(f"""
    {status_emoji} **Configuration Loaded** - {status_text}
    
    **Media Channels:** {channels_text}
    
    **Time Period:** {time_display}
    """)

#retrieve

def get_rag_context_enhanced(query, retriever):
    """Enhanced retrieval with uncertainty-focused improvements"""
    try:
        # Increase k for uncertainty queries as they might be less common
        k_value = 20 if is_uncertainty_query(query) else 15
        results = retriever.invoke(query, k=k_value)
        
        if not results:
            return "No relevant context found in the knowledge base."
        
        scored_results = score_and_filter_documents(query, results)
        return build_optimized_context(scored_results, query)
        
    except Exception as e:
        st.error(f"Error retrieving context: {str(e)}")
        return "Error retrieving relevant context."

def is_uncertainty_query(query):
    """Check if query is about uncertainty/confidence intervals"""
    uncertainty_terms = [
        'uncertainty', 'confidence', 'interval', 'reliable', 'reliability',
        'variance', 'standard error', 'credible', 'margin of error',
        'most reliable', 'least reliable', 'statistical significance'
    ]
    query_lower = query.lower()
    return any(term in query_lower for term in uncertainty_terms)

def score_and_filter_documents(query, documents):
    """Enhanced scoring with special handling for uncertainty queries"""
    query_lower = query.lower()
    query_terms = set(re.findall(r'\b\w+\b', query_lower))
    is_uncertainty = is_uncertainty_query(query)
    
    scored_docs = []
    
    for doc in documents:
        score = 0
        content_lower = doc.page_content.lower()
        
        # 1. Keyword matching (35% weight, reduced to make room for uncertainty boost)
        content_terms = set(re.findall(r'\b\w+\b', content_lower))
        if len(query_terms) > 0:
            keyword_overlap = len(query_terms.intersection(content_terms)) / len(query_terms)
            score += keyword_overlap * 0.35
        
        # 2. Category relevance (25% weight)
        category_score = get_category_relevance(query_lower, doc.metadata.get('category', ''))
        score += category_score * 0.25
        
        # 3. Content quality (20% weight)
        content_length = len(doc.page_content)
        if 200 <= content_length <= 2000:
            score += 0.2
        elif content_length > 100:
            score += 0.1
        
        # 4. Specific term boost (10% weight)
        if has_specific_terms(query_lower, content_lower):
            score += 0.1
        
        # 5. NEW: Uncertainty content boost (10% weight)
        if is_uncertainty:
            uncertainty_score = get_uncertainty_relevance(content_lower)
            score += uncertainty_score * 0.1
        
        scored_docs.append((score, doc))
    
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # For uncertainty queries, return more documents to ensure we get relevant ones
    num_docs = 8 if is_uncertainty else 5
    return scored_docs[:num_docs]

def get_uncertainty_relevance(content):
    """Score content based on uncertainty-related terms"""
    uncertainty_indicators = [
        'confidence interval', 'uncertainty', 'variance', 'standard error',
        'credible interval', 'margin of error', 'statistical significance',
        'reliability', 'precision', 'accuracy', 'robust', 'stable',
        'volatile', 'consistent', 'inconsistent', 'reliable', 'unreliable',
        'high confidence', 'low confidence', 'narrow interval', 'wide interval',
        'uncertain', 'certain', 'probability', 'likelihood'
    ]
    
    score = 0
    for indicator in uncertainty_indicators:
        if indicator in content:
            # Give higher scores for exact matches of key terms
            if indicator in ['confidence interval', 'uncertainty', 'reliability']:
                score += 0.3
            else:
                score += 0.1
    
    # Cap the score at 1.0
    return min(score, 1.0)

def has_specific_terms(query, content):
    """Enhanced to include uncertainty terms"""
    config = get_company_config()
    
    # Dynamic channel names from config
    channels = config["channels"]
    
    # Enhanced metrics including uncertainty terms
    metrics = [
        'roi', 'revenue', 'spend', 'budget', 'optimization', 'performance',
        'uncertainty', 'confidence', 'interval', 'reliable', 'reliability',
        'variance', 'error', 'margin', 'statistical', 'robust', 'stable'
    ]
    
    # Dynamic time periods from config
    time_terms = config["time_periods"]
    
    all_terms = channels + metrics + time_terms
    
    if not all_terms:
        return False
    
    for term in all_terms:
        if term in query and term in content:
            return True
    return False

def get_category_relevance(query, category):
    """Enhanced category relevance with uncertainty support"""
    config = get_company_config()
    
    category_keywords = {
        'optimization': ['optimize', 'budget', 'allocation', 'improve', 'increase'],
        'roi': ['roi', 'return', 'efficiency', 'performance', 'effective'],
        'revenue': ['revenue', 'income', 'sales', 'money', 'profit'],
        'channel': ['channel'] + config["channels"],
        'spend': ['spend', 'budget', 'cost', 'investment', 'allocation'],
        'analysis': ['analysis', 'trend', 'performance', 'compare', 'ranking'],
        'uncertainty': [
            'uncertainty', 'confidence', 'interval', 'reliable', 'reliability',
            'variance', 'error', 'margin', 'statistical', 'robust', 'stable',
            'credible', 'precision', 'accuracy', 'volatile', 'consistent'
        ]
    }
    
    category_lower = category.lower()
    
    # Direct category name matching
    for cat, keywords in category_keywords.items():
        if cat in category_lower:
            # Check for keyword matches
            for keyword in keywords:
                if keyword in query:
                    return 1.0
            # Partial match for category
            return 0.6
    
    # Special handling for uncertainty queries
    if is_uncertainty_query(query):
        uncertainty_terms_in_category = [
            'uncertainty', 'confidence', 'interval', 'error', 'variance',
            'reliable', 'robust', 'statistical'
        ]
        if any(term in category_lower for term in uncertainty_terms_in_category):
            return 1.0
    
    return 0.0

def build_optimized_context(scored_results, query):
    """Enhanced context building with uncertainty prioritization"""
    context_parts = []
    token_count = 0
    used_categories = defaultdict(int)
    is_uncertainty = is_uncertainty_query(query)
    
    # For uncertainty queries, prioritize uncertainty-related documents
    if is_uncertainty:
        uncertainty_docs = []
        other_docs = []
        
        for score, doc in scored_results:
            content_lower = doc.page_content.lower()
            category = doc.metadata.get('category', 'general').lower()
            
            if ('uncertainty' in content_lower or 'confidence' in content_lower or 
                'interval' in content_lower or 'uncertainty' in category):
                uncertainty_docs.append((score, doc))
            else:
                other_docs.append((score, doc))
        
        # Combine with uncertainty docs first
        prioritized_docs = uncertainty_docs + other_docs
    else:
        prioritized_docs = scored_results
    
    for score, doc in prioritized_docs:
        category = doc.metadata.get('category', 'general')
        
        # Allow more uncertainty documents
        max_per_category = 3 if is_uncertainty and 'uncertainty' in category.lower() else 2
        
        if used_categories[category] >= max_per_category:
            continue
        
        part = f"**{category.upper().replace('_', ' ')}:**\n{doc.page_content}"
        if doc.metadata.get('source'):
            part += f"\n*Source: {doc.metadata['source']}*"
        part += "\n"
        
        part_tokens = check_token.num_tokens_from_string(part)
        if token_count + part_tokens > (MAX_TOKENS - RESERVED_FOR_ANSWER):
            break
        
        context_parts.append(part)
        token_count += part_tokens
        used_categories[category] += 1
    
    return "\n".join(context_parts)

def get_enhanced_context(query, retriever):
    """Enhanced context with uncertainty query support"""
    rag_context = get_rag_context_enhanced(query, retriever)
    
    static_context = getattr(st.session_state, 'uploaded_context', '')
    if static_context:
        static_lines = [line for line in static_context.split('\n') 
                       if not ("CSV file" in line and "ready for analysis" in line)]
        static_context = '\n'.join(static_lines).strip()
    
    csv_analysis = analyze_csv_files_dynamically(query)
    
    sections = []
    
    if rag_context and "No relevant context found" not in rag_context:
        sections.append(f"📊 **MARKETING MIX MODEL DATA:**\n{rag_context}")
    
    if csv_analysis:
        sections.append(f"📈 **DYNAMIC DATA ANALYSIS:**\n{csv_analysis}")
    
    if static_context:
        sections.append(f"📁 **ADDITIONAL CONTEXT:**\n{static_context}")
    
    query_type = identify_query_type(query)
    if query_type:
        sections.append(f"🎯 **QUERY FOCUS:** {query_type}")
    
    return "\n\n" + "="*50 + "\n\n".join(sections)

def identify_query_type(query):
    """Enhanced query type identification with uncertainty support"""
    query_lower = query.lower()
    
    # Check for uncertainty queries first
    if is_uncertainty_query(query):
        return "Uncertainty and reliability analysis requested - focusing on confidence intervals and statistical reliability"
    elif any(word in query_lower for word in ['optimize', 'optimization', 'improve', 'better']):
        return "Optimization and improvement recommendations needed"
    elif any(word in query_lower for word in ['roi', 'return', 'efficiency', 'performance']):
        return "ROI and performance analysis requested"
    elif any(word in query_lower for word in ['compare', 'vs', 'difference', 'between']):
        return "Comparison analysis requested"
    elif any(word in query_lower for word in ['trend', 'over time', 'monthly', 'quarterly']):
        return "Trend and time-series analysis requested"
    elif any(word in query_lower for word in ['budget', 'spend', 'allocation', 'cost']):
        return "Budget and spend analysis requested"
    else:
        return "General marketing mix model inquiry"