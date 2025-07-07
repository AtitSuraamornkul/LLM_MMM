from bs4 import BeautifulSoup
import re
import json
from datetime import datetime
from collections import defaultdict
import statistics
import calendar

with open('output/new_summary_output.html', 'r', encoding='utf-8') as f:
    soup = BeautifulSoup(f, 'html.parser')


def get_channel_contribution(soup):
    import re
    import json
    
    chart_card = soup.find("card", {"id": "channel-contrib"})
    
    if not chart_card:
        return "No channel contribution data found."
    
    # Extract data (same extraction logic as before)
    insight_text = chart_card.find("p", {"class": "insights-text"})
    insight_text_content = insight_text.get_text(strip=True) if insight_text else None
    
    card_title = chart_card.find("card-title")
    card_title_content = card_title.get_text(strip=True) if card_title else None
    
    chart_description = chart_card.find("chart-description")
    chart_description_content = chart_description.get_text(strip=True) if chart_description else None
    
    # [Same JSON extraction logic as before...]
    script_tag = chart_card.find("script", {"type": "text/javascript"})
    channel_data = []
    
    if script_tag:
        script_content = script_tag.get_text()
        json_match = re.search(r'JSON\.parse\("(.+?)"\)', script_content, re.DOTALL)
        if json_match:
            try:
                escaped_json = json_match.group(1)
                unescaped_json = escaped_json.replace('\\"', '"').replace('\\n', '').replace('\\\\', '\\')
                chart_spec = json.loads(unescaped_json)
                datasets = chart_spec.get('datasets', {})
                for dataset_key, dataset_value in datasets.items():
                    if isinstance(dataset_value, list):
                        channel_data = dataset_value
                        break
            except:
                pass
    
    # Process data
    baseline_data = None
    marketing_channels = []
    
    for channel in channel_data:
        if channel.get('channel') == 'BASELINE':
            baseline_data = channel
        else:
            marketing_channels.append(channel)
    
    marketing_channels.sort(key=lambda x: x.get('incremental_outcome', 0), reverse=True)
    
    # Format for RAG/Vector DB
    baseline_pct = round(baseline_data.get('pct_of_contribution', 0) * 100, 1) if baseline_data else 0
    total_marketing_pct = round(sum(ch.get('pct_of_contribution', 0) for ch in marketing_channels) * 100, 1)
    
    # Create structured, searchable content
    rag_content = f"""
Channel Contribution Analysis:

Business Context: {insight_text_content}

Revenue Attribution:
- Baseline revenue accounts for {baseline_pct}% of total revenue
- Marketing channels drive {total_marketing_pct}% of total revenue
- Total revenue split: {baseline_pct}% organic/baseline vs {total_marketing_pct}% paid marketing

Marketing Channel Performance:
""".strip()
        
    # Add individual channel performance
    for i, channel in enumerate(marketing_channels, 1):
        ch_pct = round(channel.get('pct_of_contribution', 0) * 100, 1)
        revenue = channel.get('incremental_outcome', 0)
        rag_content += f"\n- {channel.get('channel')} contributes {ch_pct}% of total revenue (${revenue:,.0f})"
    
    # Add key insights for better retrieval
    if marketing_channels:
        top_channel = marketing_channels[0]
        top_ch_pct = round(top_channel.get('pct_of_contribution', 0) * 100, 1)
        rag_content += f"\n\nKey Insights:\n- {top_channel.get('channel')} is the top performing marketing channel at {top_ch_pct}%"
        rag_content += f"\n- Baseline/organic traffic dominates revenue generation at {baseline_pct}%"
        rag_content += f"\n- Marketing channels collectively contribute {total_marketing_pct}% to total revenue"
    
    if chart_description_content:
        rag_content += f"\n\nMethodology: {chart_description_content}"
    
    return rag_content


def get_spend_outcome_insights(soup):
    import re
    import json
    
    # Find the spend-outcome chart
    chart_embed = soup.find("chart-embed", {"id": "spend-outcome-chart"})
    if not chart_embed:
        return "No spend-outcome data found."
    
    # Find the parent chart element to get description and script
    chart_element = chart_embed.find_parent("chart")
    if not chart_element:
        return "No spend-outcome chart element found."
    
    # Extract chart description
    chart_description = chart_element.find("chart-description")
    chart_description_content = chart_description.get_text(strip=True) if chart_description else None
    
    # Extract chart data from script
    script_tag = chart_element.find_next("script", {"type": "text/javascript"})
    channel_data = []
    
    if script_tag:
        script_content = script_tag.get_text()
        
        # Find the JSON.parse() content
        json_match = re.search(r'JSON\.parse\("(.+?)"\)', script_content, re.DOTALL)
        if json_match:
            try:
                # Get the escaped JSON string and unescape it
                escaped_json = json_match.group(1)
                unescaped_json = escaped_json.replace('\\"', '"').replace('\\n', '').replace('\\\\', '\\')
                
                # Parse the JSON
                chart_spec = json.loads(unescaped_json)
                
                # Extract the dataset
                datasets = chart_spec.get('datasets', {})
                for dataset_key, dataset_value in datasets.items():
                    if isinstance(dataset_value, list):
                        channel_data = dataset_value
                        break
                        
            except (json.JSONDecodeError, AttributeError):
                pass
    
    if not channel_data:
        return "No spend-outcome data could be extracted."
    
    # Process the data - separate revenue and spend data
    revenue_data = [ch for ch in channel_data if ch.get('label') == '% Revenue']
    spend_data = [ch for ch in channel_data if ch.get('label') == '% Spend']
    
    # Create channel analysis by combining revenue and spend data
    channel_analysis = {}
    
    for rev_ch in revenue_data:
        channel = rev_ch['channel']
        spend_ch = next((s for s in spend_data if s['channel'] == channel), None)
        
        if spend_ch:
            channel_analysis[channel] = {
                'revenue_pct': rev_ch['pct'] * 100,
                'spend_pct': spend_ch['pct'] * 100,
                'roi': rev_ch['roi'],
                'efficiency': rev_ch['pct'] / spend_ch['pct'] if spend_ch['pct'] > 0 else 0
            }
    
    # Sort channels by ROI
    sorted_channels = sorted(channel_analysis.items(), key=lambda x: x[1]['roi'], reverse=True)
    
    # Calculate totals
    total_revenue_pct = sum(ch['revenue_pct'] for ch in channel_analysis.values())
    total_spend_pct = sum(ch['spend_pct'] for ch in channel_analysis.values())
    avg_roi = sum(ch['roi'] for ch in channel_analysis.values()) / len(channel_analysis)
    
    # Format results for RAG
    rag_content = f"""
Marketing Channel Spend and ROI Analysis:

Performance Overview:
- Marketing channels account for {total_revenue_pct:.1f}% of attributed revenue
- Total marketing spend allocation: {total_spend_pct:.1f}%
- Average ROI across all channels: {avg_roi:.1f}x

Channel Performance by ROI:
""".strip()
    
    # Add individual channel performance
    for channel, data in sorted_channels:
        rag_content += f"\n- {channel.upper()}: {data['roi']:.1f}x ROI, {data['revenue_pct']:.1f}% revenue share, {data['spend_pct']:.1f}% spend share"
    
    # Add efficiency insights
    rag_content += f"\n\nChannel Efficiency Analysis:"
    most_efficient = max(sorted_channels, key=lambda x: x[1]['efficiency'])
    least_efficient = min(sorted_channels, key=lambda x: x[1]['efficiency'])
    
    rag_content += f"\n- Most efficient spend allocation: {most_efficient[0].upper()} (revenue/spend ratio: {most_efficient[1]['efficiency']:.2f})"
    rag_content += f"\n- Least efficient spend allocation: {least_efficient[0].upper()} (revenue/spend ratio: {least_efficient[1]['efficiency']:.2f})"
    
    # ROI insights
    best_roi_channel = sorted_channels[0]
    worst_roi_channel = sorted_channels[-1]
    
    rag_content += f"\n\nROI Performance:"
    rag_content += f"\n- Highest ROI: {best_roi_channel[0].upper()} at {best_roi_channel[1]['roi']:.1f}x return"
    rag_content += f"\n- Lowest ROI: {worst_roi_channel[0].upper()} at {worst_roi_channel[1]['roi']:.1f}x return"
    rag_content += f"\n- ROI range: {worst_roi_channel[1]['roi']:.1f}x to {best_roi_channel[1]['roi']:.1f}x across all channels"
    
    # Budget allocation insights
    rag_content += f"\n\nBudget Allocation Insights:"
    for channel, data in sorted_channels:
        if data['revenue_pct'] > data['spend_pct']:
            rag_content += f"\n- {channel.upper()}: Over-performing (generates {data['revenue_pct']:.1f}% revenue with {data['spend_pct']:.1f}% spend)"
        elif data['revenue_pct'] < data['spend_pct']:
            rag_content += f"\n- {channel.upper()}: Under-performing (generates {data['revenue_pct']:.1f}% revenue with {data['spend_pct']:.1f}% spend)"
        else:
            rag_content += f"\n- {channel.upper()}: Balanced performance (revenue and spend percentages aligned)"
    
    if chart_description_content:
        rag_content += f"\n\nMethodology: {chart_description_content}"

    return rag_content



def get_channel_time_insights_with_anomalies(soup):
    
    # [Same extraction logic as before...]
    chart_embed = soup.find("chart-embed", {"id": "channel-contrib-by-time-chart"})
    if not chart_embed:
        return "No time-series channel contribution data found."
    
    chart_element = chart_embed.find_parent("chart")
    if not chart_element:
        return "No time-series chart element found."
    
    chart_description = chart_element.find("chart-description")
    chart_description_content = chart_description.get_text(strip=True) if chart_description else None
    
    script_tag = chart_element.find_next("script", {"type": "text/javascript"})
    time_data = []
    
    if script_tag:
        script_content = script_tag.get_text()
        json_match = re.search(r'JSON\.parse\("(.+?)"\)', script_content, re.DOTALL)
        if json_match:
            try:
                escaped_json = json_match.group(1)
                unescaped_json = escaped_json.replace('\\"', '"').replace('\\n', '').replace('\\\\', '\\')
                chart_spec = json.loads(unescaped_json)
                datasets = chart_spec.get('datasets', {})
                for dataset_key, dataset_value in datasets.items():
                    if isinstance(dataset_value, list):
                        time_data = dataset_value
                        break
            except (json.JSONDecodeError, AttributeError):
                pass
    
    if not time_data:
        return "No time-series data could be extracted."
    
    # Organize data by channel and time
    channel_monthly_data = defaultdict(lambda: defaultdict(dict))
    channel_quarterly_data = defaultdict(lambda: defaultdict(dict))
    
    for record in time_data:
        try:
            date_obj = datetime.strptime(record['time'], '%Y-%m-%d')
            revenue = record.get('incremental_outcome', 0) or 0
            contribution_pct = record.get('pct_of_contribution', 0) * 100
            channel = record['channel']
            
            # Monthly data by channel
            month_key = date_obj.strftime('%Y-%m')
            channel_monthly_data[channel][month_key] = {
                'revenue': revenue,
                'contribution_pct': contribution_pct,
                'date': date_obj
            }
            
            # Quarterly data by channel
            quarter = f"Q{(date_obj.month - 1) // 3 + 1}"
            quarter_key = f"{date_obj.year}-{quarter}"
            
            # Keep the latest data point for each quarter
            if (quarter_key not in channel_quarterly_data[channel] or 
                date_obj > channel_quarterly_data[channel][quarter_key].get('date', datetime.min)):
                channel_quarterly_data[channel][quarter_key] = {
                    'revenue': revenue,
                    'contribution_pct': contribution_pct,
                    'date': date_obj
                }
                
        except (ValueError, KeyError):
            continue
    
    # Generate comprehensive summaries
    chunks = []
    
    # Chunk 1: Monthly Performance by Channel with Anomalies
    chunk1 = create_monthly_channel_summaries_with_anomalies(channel_monthly_data)
    chunks.append(chunk1)
    
    # Chunk 2: Quarterly Performance by Channel with Trends
    chunk2 = create_quarterly_channel_summaries_with_trends(channel_quarterly_data)
    chunks.append(chunk2)
    
    # Chunk 3: Spike and Dip Analysis Across All Channels
    chunk3 = create_anomaly_analysis(channel_monthly_data)
    chunks.append(chunk3)
    
    # Chunk 4: Channel Performance Comparison and Rankings
    chunk4 = create_channel_comparison_analysis(channel_monthly_data, channel_quarterly_data)
    chunks.append(chunk4)
    
    # Chunk 5: Growth Momentum and Trend Analysis
    chunk5 = create_momentum_analysis(channel_monthly_data, channel_quarterly_data)
    chunks.append(chunk5)
    
    # Create formatted output
    formatted_output = []
    
    formatted_output.append("=== MONTHLY CHANNEL PERFORMANCE WITH ANOMALIES ===")
    formatted_output.append(chunks[0])
    formatted_output.append("\n=== QUARTERLY CHANNEL TRENDS ===")
    formatted_output.append(chunks[1])
    formatted_output.append("\n=== SPIKE AND DIP ANALYSIS ===")
    formatted_output.append(chunks[2])
    formatted_output.append("\n=== CHANNEL COMPARISON ===")
    formatted_output.append(chunks[3])
    formatted_output.append("\n=== MOMENTUM ANALYSIS ===")
    formatted_output.append(chunks[4])
    
    # Return both individual chunks and formatted output
    return {
        'chunks': chunks,
        'formatted_output': '\n'.join(formatted_output),
        'summary_sections': {
            'monthly_anomalies': chunks[0],
            'quarterly_trends': chunks[1],
            'spike_dip_analysis': chunks[2],
            'channel_comparison': chunks[3],
            'momentum_analysis': chunks[4]
        }
    }

def detect_spikes_and_dips(revenue_series, threshold_multiplier=1.5):
    """Detect spikes and dips in revenue series"""
    if len(revenue_series) < 3:
        return [], []
    
    # Calculate moving average and standard deviation
    values = [v for v in revenue_series.values() if v > 0]
    if len(values) < 2:
        return [], []
    
    mean_revenue = statistics.mean(values)
    std_revenue = statistics.stdev(values) if len(values) > 1 else 0
    
    spikes = []
    dips = []
    
    for period, revenue in revenue_series.items():
        if revenue > 0:
            # Spike detection: revenue > mean + (threshold * std)
            if revenue > mean_revenue + (threshold_multiplier * std_revenue):
                spike_magnitude = ((revenue - mean_revenue) / mean_revenue) * 100
                spikes.append({
                    'period': period,
                    'revenue': revenue,
                    'magnitude': spike_magnitude
                })
            
            # Dip detection: revenue < mean - (threshold * std) and significantly below mean
            elif revenue < mean_revenue - (threshold_multiplier * std_revenue) and revenue < mean_revenue * 0.5:
                dip_magnitude = ((mean_revenue - revenue) / mean_revenue) * 100
                dips.append({
                    'period': period,
                    'revenue': revenue,
                    'magnitude': dip_magnitude
                })
    
    return spikes, dips

def create_monthly_channel_summaries_with_anomalies(channel_monthly_data):
    """Create monthly summaries by channel with spike/dip detection"""
    
    summary = "Monthly Channel Performance Analysis with Anomaly Detection:\n"
    
    for channel in sorted(channel_monthly_data.keys()):
        if channel == 'baseline':
            continue
            
        monthly_data = channel_monthly_data[channel]
        if not monthly_data:
            continue
        
        # Get revenue series for spike/dip detection
        revenue_series = {month: data['revenue'] for month, data in monthly_data.items()}
        spikes, dips = detect_spikes_and_dips(revenue_series)
        
        # Calculate basic stats
        revenues = [data['revenue'] for data in monthly_data.values() if data['revenue'] > 0]
        if not revenues:
            continue
            
        avg_revenue = statistics.mean(revenues)
        max_revenue = max(revenues)
        min_revenue = min(revenues)
        
        # Find peak and trough months
        peak_month = max(monthly_data.items(), key=lambda x: x[1]['revenue'])
        trough_month = min(monthly_data.items(), key=lambda x: x[1]['revenue'])
        
        summary += f"""
{channel.upper()} - Monthly Performance:
- Average Monthly Revenue: ${avg_revenue:,.0f}
- Peak Month: {peak_month[0]} (${peak_month[1]['revenue']:,.0f})
- Lowest Month: {trough_month[0]} (${trough_month[1]['revenue']:,.0f})
- Active Months: {len([r for r in revenues if r > 0])} out of {len(monthly_data)}
"""
        
        # Add spike analysis
        if spikes:
            summary += f"- Revenue Spikes Detected: {len(spikes)}\n"
            for spike in sorted(spikes, key=lambda x: x['magnitude'], reverse=True)[:3]:
                summary += f"  • {spike['period']}: ${spike['revenue']:,.0f} (+{spike['magnitude']:.0f}% above average)\n"
        
        # Add dip analysis
        if dips:
            summary += f"- Revenue Dips Detected: {len(dips)}\n"
            for dip in sorted(dips, key=lambda x: x['magnitude'], reverse=True)[:3]:
                summary += f"  • {dip['period']}: ${dip['revenue']:,.0f} (-{dip['magnitude']:.0f}% below average)\n"
        
        # Month-over-month growth analysis
        sorted_months = sorted(monthly_data.keys())
        growth_periods = []
        decline_periods = []
        
        for i in range(1, len(sorted_months)):
            current_month = sorted_months[i]
            prev_month = sorted_months[i-1]
            
            current_rev = monthly_data[current_month]['revenue']
            prev_rev = monthly_data[prev_month]['revenue']
            
            if prev_rev > 0 and current_rev > 0:
                growth_rate = ((current_rev - prev_rev) / prev_rev) * 100
                if growth_rate > 50:  # Significant growth
                    growth_periods.append(f"{current_month} (+{growth_rate:.0f}%)")
                elif growth_rate < -50:  # Significant decline
                    decline_periods.append(f"{current_month} ({growth_rate:.0f}%)")
        
        if growth_periods:
            summary += f"- High Growth Periods: {', '.join(growth_periods[:3])}\n"
        if decline_periods:
            summary += f"- Decline Periods: {', '.join(decline_periods[:3])}\n"
        
        summary += "\n"
    
    return summary.strip()

def create_quarterly_channel_summaries_with_trends(channel_quarterly_data):
    """Create quarterly summaries by channel with trend analysis"""
    
    summary = "Quarterly Channel Performance Analysis with Trend Detection:\n"
    
    for channel in sorted(channel_quarterly_data.keys()):
        if channel == 'baseline':
            continue
            
        quarterly_data = channel_quarterly_data[channel]
        if not quarterly_data:
            continue
        
        sorted_quarters = sorted(quarterly_data.keys())
        if len(sorted_quarters) < 2:
            continue
        
        # Calculate quarterly trends
        revenues = [quarterly_data[q]['revenue'] for q in sorted_quarters]
        
        # Trend analysis
        trend_direction = "stable"
        if len(revenues) >= 3:
            recent_trend = revenues[-3:]
            if all(recent_trend[i] < recent_trend[i+1] for i in range(len(recent_trend)-1)):
                trend_direction = "consistently growing"
            elif all(recent_trend[i] > recent_trend[i+1] for i in range(len(recent_trend)-1)):
                trend_direction = "consistently declining"
            elif revenues[-1] > revenues[0]:
                trend_direction = "overall growth"
            elif revenues[-1] < revenues[0]:
                trend_direction = "overall decline"
        
        # Quarter-over-quarter growth rates
        qoq_growth = []
        for i in range(1, len(sorted_quarters)):
            current_rev = quarterly_data[sorted_quarters[i]]['revenue']
            prev_rev = quarterly_data[sorted_quarters[i-1]]['revenue']
            
            if prev_rev > 0:
                growth_rate = ((current_rev - prev_rev) / prev_rev) * 100
                qoq_growth.append((sorted_quarters[i], growth_rate))
        
        # Find best and worst quarters
        best_quarter = max(quarterly_data.items(), key=lambda x: x[1]['revenue'])
        worst_quarter = min(quarterly_data.items(), key=lambda x: x[1]['revenue'])
        
        summary += f"""
{channel.upper()} - Quarterly Analysis:
- Trend Direction: {trend_direction.title()}
- Best Quarter: {best_quarter[0]} (${best_quarter[1]['revenue']:,.0f})
- Worst Quarter: {worst_quarter[0]} (${worst_quarter[1]['revenue']:,.0f})
- Total Quarters Active: {len([r for r in revenues if r > 0])}
"""
        
        # Add significant QoQ changes
        significant_growth = [q for q, g in qoq_growth if g > 100]
        significant_decline = [q for q, g in qoq_growth if g < -50]
        
        if significant_growth:
            summary += f"- Major Growth Quarters: {', '.join(significant_growth)}\n"
        if significant_decline:
            summary += f"- Major Decline Quarters: {', '.join(significant_decline)}\n"
        
        # Recent performance (last 2 quarters)
        if len(sorted_quarters) >= 2:
            recent_quarters = sorted_quarters[-2:]
            recent_avg = statistics.mean([quarterly_data[q]['revenue'] for q in recent_quarters])
            summary += f"- Recent Performance (Last 2Q): ${recent_avg:,.0f} average\n"
        
        summary += "\n"
    
    return summary.strip()

def create_anomaly_analysis(channel_monthly_data):
    """Create comprehensive spike and dip analysis across all channels"""
    
    summary = "Revenue Anomaly Analysis - Spikes and Dips Across All Channels:\n"
    
    all_spikes = []
    all_dips = []
    
    # Collect all spikes and dips across channels
    for channel, monthly_data in channel_monthly_data.items():
        if channel == 'baseline':
            continue
            
        revenue_series = {month: data['revenue'] for month, data in monthly_data.items()}
        spikes, dips = detect_spikes_and_dips(revenue_series)
        
        for spike in spikes:
            spike['channel'] = channel
            all_spikes.append(spike)
            
        for dip in dips:
            dip['channel'] = channel
            all_dips.append(dip)
    
    # Analyze biggest spikes
    if all_spikes:
        summary += f"\nTop Revenue Spikes (Highest Magnitude):\n"
        top_spikes = sorted(all_spikes, key=lambda x: x['magnitude'], reverse=True)[:5]
        for i, spike in enumerate(top_spikes, 1):
            summary += f"{i}. {spike['channel'].upper()} in {spike['period']}: ${spike['revenue']:,.0f} (+{spike['magnitude']:.0f}% above average)\n"
    
    # Analyze biggest dips
    if all_dips:
        summary += f"\nSignificant Revenue Dips:\n"
        top_dips = sorted(all_dips, key=lambda x: x['magnitude'], reverse=True)[:5]
        for i, dip in enumerate(top_dips, 1):
            summary += f"{i}. {dip['channel'].upper()} in {dip['period']}: ${dip['revenue']:,.0f} (-{dip['magnitude']:.0f}% below average)\n"
    
    # Seasonal spike analysis
    spike_months = defaultdict(int)
    dip_months = defaultdict(int)
    
    for spike in all_spikes:
        try:
            month_num = int(spike['period'].split('-')[1])
            spike_months[month_num] += 1
        except:
            pass
    
    for dip in all_dips:
        try:
            month_num = int(dip['period'].split('-')[1])
            dip_months[month_num] += 1
        except:
            pass
    
    if spike_months:
        peak_spike_month = max(spike_months.items(), key=lambda x: x[1])
        month_name = calendar.month_name[peak_spike_month[0]]
        summary += f"\nSeasonal Patterns:\n"
        summary += f"- Most Common Spike Month: {month_name} ({peak_spike_month[1]} occurrences)\n"
    
    if dip_months:
        peak_dip_month = max(dip_months.items(), key=lambda x: x[1])
        month_name = calendar.month_name[peak_dip_month[0]]
        summary += f"- Most Common Dip Month: {month_name} ({peak_dip_month[1]} occurrences)\n"
    
    # Channel volatility analysis
    channel_volatility = {}
    for channel, monthly_data in channel_monthly_data.items():
        if channel == 'baseline':
            continue
            
        revenues = [data['revenue'] for data in monthly_data.values() if data['revenue'] > 0]
        if len(revenues) > 1:
            avg_rev = statistics.mean(revenues)
            std_rev = statistics.stdev(revenues)
            volatility = (std_rev / avg_rev) * 100 if avg_rev > 0 else 0
            channel_volatility[channel] = volatility
    
    if channel_volatility:
        summary += f"\nChannel Volatility Analysis (Revenue Consistency):\n"
        sorted_volatility = sorted(channel_volatility.items(), key=lambda x: x[1])
        
        most_stable = sorted_volatility[0]
        most_volatile = sorted_volatility[-1]
        
        summary += f"- Most Stable Channel: {most_stable[0].upper()} ({most_stable[1]:.0f}% volatility)\n"
        summary += f"- Most Volatile Channel: {most_volatile[0].upper()} ({most_volatile[1]:.0f}% volatility)\n"
    
    return summary.strip()

def create_channel_comparison_analysis(channel_monthly_data, channel_quarterly_data):
    """Create channel performance comparison and rankings"""
    
    summary = "Channel Performance Comparison and Rankings:\n"
    
    # Calculate total revenue by channel
    channel_totals = {}
    channel_consistency = {}
    channel_peak_performance = {}
    
    for channel, monthly_data in channel_monthly_data.items():
        if channel == 'baseline':
            continue
            
        revenues = [data['revenue'] for data in monthly_data.values()]
        active_revenues = [r for r in revenues if r > 0]
        
        if active_revenues:
            channel_totals[channel] = sum(active_revenues)
            channel_consistency[channel] = len(active_revenues) / len(monthly_data) * 100
            channel_peak_performance[channel] = max(active_revenues)
    
    # Rankings
    if channel_totals:
        summary += f"\nTotal Revenue Rankings:\n"
        revenue_ranking = sorted(channel_totals.items(), key=lambda x: x[1], reverse=True)
        for i, (channel, total_rev) in enumerate(revenue_ranking, 1):
            summary += f"{i}. {channel.upper()}: ${total_rev:,.0f} cumulative revenue\n"
    
    if channel_consistency:
        summary += f"\nConsistency Rankings (% of months active):\n"
        consistency_ranking = sorted(channel_consistency.items(), key=lambda x: x[1], reverse=True)
        for i, (channel, consistency) in enumerate(consistency_ranking, 1):
            summary += f"{i}. {channel.upper()}: {consistency:.0f}% of months with revenue\n"
    
    if channel_peak_performance:
        summary += f"\nPeak Performance Rankings (Highest single month):\n"
        peak_ranking = sorted(channel_peak_performance.items(), key=lambda x: x[1], reverse=True)
        for i, (channel, peak_rev) in enumerate(peak_ranking, 1):
            summary += f"{i}. {channel.upper()}: ${peak_rev:,.0f} peak monthly revenue\n"
    
    # Channel lifecycle analysis
    summary += f"\nChannel Lifecycle Analysis:\n"
    
    for channel, monthly_data in channel_monthly_data.items():
        if channel == 'baseline':
            continue
            
        sorted_months = sorted(monthly_data.keys())
        active_months = [month for month in sorted_months if monthly_data[month]['revenue'] > 0]
        
        if active_months:
            launch_month = active_months[0]
            latest_month = active_months[-1]
            
            # Determine lifecycle stage
            recent_months = sorted_months[-3:] if len(sorted_months) >= 3 else sorted_months
            recent_activity = sum(1 for month in recent_months if monthly_data[month]['revenue'] > 0)
            
            if recent_activity == 0:
                stage = "Dormant"
            elif len(active_months) <= 3:
                stage = "Launch Phase"
            elif recent_activity == len(recent_months):
                stage = "Active/Mature"
            else:
                stage = "Intermittent"
            
            summary += f"- {channel.upper()}: {stage} (Active: {launch_month} to {latest_month})\n"
    
    return summary.strip()

def create_momentum_analysis(channel_monthly_data, channel_quarterly_data):
    """Create growth momentum and trend analysis"""
    
    summary = "Channel Growth Momentum and Trend Analysis:\n"
    
    for channel, monthly_data in channel_monthly_data.items():
        if channel == 'baseline':
            continue
            
        sorted_months = sorted(monthly_data.keys())
        if len(sorted_months) < 6:  # Need at least 6 months for momentum analysis
            continue
        
        # Get last 6 months of data
        recent_months = sorted_months[-6:]
        recent_revenues = [monthly_data[month]['revenue'] for month in recent_months]
        
        # Calculate momentum indicators
        first_half_avg = statistics.mean(recent_revenues[:3])
        second_half_avg = statistics.mean(recent_revenues[3:])
        
        momentum = "neutral"
        momentum_pct = 0
        
        if first_half_avg > 0:
            momentum_pct = ((second_half_avg - first_half_avg) / first_half_avg) * 100
            
            if momentum_pct > 20:
                momentum = "strong positive"
            elif momentum_pct > 5:
                momentum = "positive"
            elif momentum_pct < -20:
                momentum = "strong negative"
            elif momentum_pct < -5:
                momentum = "negative"
        
        # Trend consistency
        positive_months = sum(1 for i in range(1, len(recent_revenues)) 
                             if recent_revenues[i] > recent_revenues[i-1])
        trend_consistency = (positive_months / (len(recent_revenues) - 1)) * 100
        
        summary += f"""
{channel.upper()} - 6-Month Momentum Analysis:
- Growth Momentum: {momentum.title()} ({momentum_pct:+.0f}%)
- Trend Consistency: {trend_consistency:.0f}% of months showed growth
- Recent Average (Last 3 months): ${second_half_avg:,.0f}
- Previous Average (3 months prior): ${first_half_avg:,.0f}
"""
        
    
        
        summary += "\n"
    
    return summary.strip()


def get_roi_insights(soup):
    """Extract ROI title and insights text"""
    
    # Find the ROI card
    roi_card = soup.find("card", {"id": "performance-breakdown"})
    if not roi_card:
        return "No ROI performance data found."
    
    # Extract card title
    card_title = roi_card.find("card-title")
    card_title_content = card_title.get_text(strip=True) if card_title else None
    
    # Extract insights text
    insights_text = roi_card.find("p", {"class": "insights-text"})
    insights_text_content = insights_text.get_text(strip=True) if insights_text else None
    
    if not card_title_content and not insights_text_content:
        return "No ROI data found."
    
    # Format output
    output = ""
    if card_title_content:
        output += f"Title: {card_title_content}\n\n"
    
    if insights_text_content:
        output += f"Insights: {insights_text_content}"
    
    return output


def get_roi_effectiveness_insights(soup):
    """Extract ROI vs Effectiveness chart insights"""
    import re
    import json
    
    # Find the ROI effectiveness chart
    chart_embed = soup.find("chart-embed", {"id": "roi-effectiveness-chart"})
    if not chart_embed:
        return "No ROI effectiveness chart found."
    
    # Find the parent chart element
    chart_element = chart_embed.find_parent("chart")
    if not chart_element:
        return "No ROI effectiveness chart element found."
    
    # Extract chart description
    chart_description = chart_element.find("chart-description")
    chart_description_content = chart_description.get_text(strip=True) if chart_description else None
    
    # Extract chart data from script
    script_tag = chart_element.find_next("script", {"type": "text/javascript"})
    chart_data = []
    
    if script_tag:
        script_content = script_tag.get_text()
        json_match = re.search(r'JSON\.parse\("(.+?)"\)', script_content, re.DOTALL)
        if json_match:
            try:
                escaped_json = json_match.group(1)
                unescaped_json = escaped_json.replace('\\"', '"').replace('\\n', '').replace('\\\\', '\\')
                chart_spec = json.loads(unescaped_json)
                
                datasets = chart_spec.get('datasets', {})
                for dataset_key, dataset_value in datasets.items():
                    if isinstance(dataset_value, list):
                        chart_data = dataset_value
                        break
            except (json.JSONDecodeError, AttributeError):
                pass
    
    if not chart_data:
        return "No ROI effectiveness data could be extracted."
    
    # Analyze the data
    analysis = analyze_roi_effectiveness(chart_data, chart_description_content)
    
    return analysis

def analyze_roi_effectiveness(chart_data, chart_description):
    """Analyze ROI vs Effectiveness data"""
    
    # Sort channels by different metrics
    by_roi = sorted(chart_data, key=lambda x: x['roi'], reverse=True)
    by_effectiveness = sorted(chart_data, key=lambda x: x['effectiveness'], reverse=True)
    by_spend = sorted(chart_data, key=lambda x: x['spend'], reverse=True)
    
    # Calculate totals and averages
    total_spend = sum(ch['spend'] for ch in chart_data)
    avg_roi = sum(ch['roi'] for ch in chart_data) / len(chart_data)
    avg_effectiveness = sum(ch['effectiveness'] for ch in chart_data) / len(chart_data)
    
    # Categorize channels based on ROI and effectiveness
    high_roi_threshold = avg_roi
    high_effectiveness_threshold = avg_effectiveness
    
    channel_categories = {}
    for channel in chart_data:
        roi_level = "High" if channel['roi'] >= high_roi_threshold else "Low"
        eff_level = "High" if channel['effectiveness'] >= high_effectiveness_threshold else "Low"
        
        if roi_level == "High" and eff_level == "High":
            category = "Star Performers"
        elif roi_level == "High" and eff_level == "Low":
            category = "Cost Efficient"
        elif roi_level == "Low" and eff_level == "High":
            category = "High Potential"
        else:
            category = "Optimization Needed"
        
        channel_categories[channel['channel']] = {
            'category': category,
            'roi': channel['roi'],
            'effectiveness': channel['effectiveness'],
            'spend': channel['spend'],
            'spend_share': (channel['spend'] / total_spend) * 100
        }
    
    # Build analysis
    analysis = f"""
ROI vs Effectiveness Analysis:

Performance Overview:
- Average ROI across channels: {avg_roi:.1f}x
- Average effectiveness: {avg_effectiveness:.4f} incremental outcome per impression
- Total media spend analyzed: ${total_spend:,.0f}

ROI Rankings:
""".strip()
    
    for i, channel in enumerate(by_roi, 1):
        analysis += f"\n{i}. {channel['channel'].upper()}: {channel['roi']:.1f}x ROI"
    
    analysis += f"\n\nEffectiveness Rankings:"
    for i, channel in enumerate(by_effectiveness, 1):
        analysis += f"\n{i}. {channel['channel'].upper()}: {channel['effectiveness']:.4f} outcome per impression"
    
    analysis += f"\n\nSpend Allocation:"
    for i, channel in enumerate(by_spend, 1):
        spend_pct = (channel['spend'] / total_spend) * 100
        analysis += f"\n{i}. {channel['channel'].upper()}: ${channel['spend']:,.0f} ({spend_pct:.1f}% of total spend)"
    
    # Channel categorization
    analysis += f"\n\nChannel Performance Categories:"
    
    categories = {}
    for channel, data in channel_categories.items():
        category = data['category']
        if category not in categories:
            categories[category] = []
        categories[category].append(f"{channel.upper()} ({data['roi']:.1f}x ROI, {data['effectiveness']:.4f} effectiveness)")
    
    for category, channels in categories.items():
        analysis += f"\n\n{category}:"
        for channel_info in channels:
            analysis += f"\n- {channel_info}"
    
    # Strategic insights
    analysis += f"\n\nStrategic Insights:"
    
    # Find best performers
    highest_roi = by_roi[0]
    highest_effectiveness = by_effectiveness[0]
    largest_spend = by_spend[0]
    
    analysis += f"\n- {highest_roi['channel'].upper()} delivers highest ROI ({highest_roi['roi']:.1f}x) - prioritize for budget allocation"
    analysis += f"\n- {highest_effectiveness['channel'].upper()} shows highest effectiveness ({highest_effectiveness['effectiveness']:.4f}) - strong media performance per impression"
    analysis += f"\n- {largest_spend['channel'].upper()} receives largest budget (${largest_spend['spend']:,.0f}) - monitor efficiency closely"
    
    # Efficiency vs spend analysis
    for channel, data in channel_categories.items():
        if data['category'] == "Star Performers":
            analysis += f"\n- {channel.upper()}: Ideal performance - high ROI and effectiveness, maintain investment"
        elif data['category'] == "Cost Efficient":
            analysis += f"\n- {channel.upper()}: Cost efficient but low reach - consider scaling if effectiveness can be maintained"
        elif data['category'] == "High Potential":
            analysis += f"\n- {channel.upper()}: High effectiveness but expensive - optimize costs to improve ROI"
        elif data['category'] == "Optimization Needed":
            analysis += f"\n- {channel.upper()}: Both ROI and effectiveness below average - requires optimization or budget reallocation"
    
    if chart_description:
        analysis += f"\n\nMethodology: {chart_description}"
    
    return analysis



def get_roi_marginal_insights(soup):
    """Extract ROI vs Marginal ROI chart insights - focused on performance analysis"""
    import re
    import json
    
    # Find the ROI marginal chart
    chart_embed = soup.find("chart-embed", {"id": "roi-marginal-chart"})
    if not chart_embed:
        return "No ROI marginal chart found."
    
    # Find the parent chart element
    chart_element = chart_embed.find_parent("chart")
    if not chart_element:
        return "No ROI marginal chart element found."
    
    # Extract chart description
    chart_description = chart_element.find("chart-description")
    chart_description_content = chart_description.get_text(strip=True) if chart_description else None
    
    # Extract chart data from script
    script_tag = chart_element.find_next("script", {"type": "text/javascript"})
    chart_data = []
    
    if script_tag:
        script_content = script_tag.get_text()
        json_match = re.search(r'JSON\.parse\("(.+?)"\)', script_content, re.DOTALL)
        if json_match:
            try:
                escaped_json = json_match.group(1)
                unescaped_json = escaped_json.replace('\\"', '"').replace('\\n', '').replace('\\\\', '\\')
                chart_spec = json.loads(unescaped_json)
                
                datasets = chart_spec.get('datasets', {})
                for dataset_key, dataset_value in datasets.items():
                    if isinstance(dataset_value, list):
                        chart_data = dataset_value
                        break
            except (json.JSONDecodeError, AttributeError):
                pass
    
    if not chart_data:
        return "No ROI marginal data could be extracted."
    
    # Analyze the data
    analysis = analyze_roi_marginal_performance(chart_data, chart_description_content)
    
    return analysis

def analyze_roi_marginal_performance(chart_data, chart_description):
    """Analyze ROI vs Marginal ROI performance insights only"""
    
    # Sort channels by different metrics
    by_roi = sorted(chart_data, key=lambda x: x['roi'], reverse=True)
    by_marginal_roi = sorted(chart_data, key=lambda x: x['mroi'], reverse=True)
    by_spend = sorted(chart_data, key=lambda x: x['spend'], reverse=True)
    
    # Calculate totals and averages
    total_spend = sum(ch['spend'] for ch in chart_data)
    avg_roi = sum(ch['roi'] for ch in chart_data) / len(chart_data)
    avg_marginal_roi = sum(ch['mroi'] for ch in chart_data) / len(chart_data)
    
    # Build analysis
    analysis = f"""
ROI vs Marginal ROI Performance Analysis:

Performance Metrics:
- Average ROI across channels: {avg_roi:.1f}x
- Average Marginal ROI: {avg_marginal_roi:.1f}x (additional return per additional dollar)
- Total media spend analyzed: ${total_spend:,.0f}

ROI Rankings:
""".strip()
    
    for i, channel in enumerate(by_roi, 1):
        analysis += f"\n{i}. {channel['channel'].upper()}: {channel['roi']:.1f}x ROI"
    
    analysis += f"\n\nMarginal ROI Rankings (Incremental Efficiency):"
    for i, channel in enumerate(by_marginal_roi, 1):
        analysis += f"\n{i}. {channel['channel'].upper()}: {channel['mroi']:.1f}x marginal ROI"
    
    analysis += f"\n\nSpend Distribution:"
    for i, channel in enumerate(by_spend, 1):
        spend_pct = (channel['spend'] / total_spend) * 100
        analysis += f"\n{i}. {channel['channel'].upper()}: ${channel['spend']:,.0f} ({spend_pct:.1f}% of total spend)"
    
    # Saturation analysis (key insight for MMM)
    analysis += f"\n\nSaturation Indicators:"
    for channel in chart_data:
        roi_marginal_ratio = channel['roi'] / channel['mroi']
        saturation_level = "High" if roi_marginal_ratio > 2.5 else "Moderate" if roi_marginal_ratio > 2.0 else "Low"
        analysis += f"\n- {channel['channel'].upper()}: {saturation_level} saturation signal (ROI/Marginal ROI ratio: {roi_marginal_ratio:.1f}x)"
    
    # Efficiency gap analysis
    analysis += f"\n\nEfficiency Gap Analysis:"
    for channel in chart_data:
        efficiency_gap = channel['roi'] - channel['mroi']
        analysis += f"\n- {channel['channel'].upper()}: {efficiency_gap:.1f}x gap between current and marginal returns"
    
    # Performance consistency analysis
    analysis += f"\n\nPerformance Consistency:"
    roi_std = (sum((ch['roi'] - avg_roi) ** 2 for ch in chart_data) / len(chart_data)) ** 0.5
    marginal_roi_std = (sum((ch['mroi'] - avg_marginal_roi) ** 2 for ch in chart_data) / len(chart_data)) ** 0.5
    
    analysis += f"\n- ROI variation across channels: {roi_std:.2f} standard deviation"
    analysis += f"\n- Marginal ROI variation: {marginal_roi_std:.2f} standard deviation"
    
    if marginal_roi_std > roi_std:
        analysis += f"\n- Marginal ROI shows higher variation than base ROI, indicating different scaling potentials"
    else:
        analysis += f"\n- ROI and marginal ROI show similar variation patterns"
    
    # Channel performance patterns
    analysis += f"\n\nChannel Performance Patterns:"
    
    best_roi = by_roi[0]
    best_marginal = by_marginal_roi[0]
    largest_spend = by_spend[0]
    
    if best_roi['channel'] == best_marginal['channel']:
        analysis += f"\n- {best_roi['channel'].upper()} leads in both ROI and marginal ROI - consistent high performer"
    else:
        analysis += f"\n- {best_roi['channel'].upper()} has highest ROI while {best_marginal['channel'].upper()} has highest marginal ROI - different optimization opportunities"
    
    # Spend efficiency vs performance
    for channel in chart_data:
        spend_share = (channel['spend'] / total_spend) * 100
        if spend_share > 50 and channel['mroi'] < avg_marginal_roi:
            analysis += f"\n- {channel['channel'].upper()}: High spend concentration ({spend_share:.1f}%) with below-average marginal efficiency"
        elif spend_share < 15 and channel['mroi'] > avg_marginal_roi:
            analysis += f"\n- {channel['channel'].upper()}: Low spend allocation ({spend_share:.1f}%) but above-average marginal efficiency"
    
    # Diminishing returns analysis
    analysis += f"\n\nDiminishing Returns Assessment:"
    for channel in chart_data:
        returns_ratio = channel['roi'] / channel['mroi']
        if returns_ratio > 3.0:
            analysis += f"\n- {channel['channel'].upper()}: Strong diminishing returns pattern (current ROI {returns_ratio:.1f}x higher than marginal)"
        elif returns_ratio > 2.0:
            analysis += f"\n- {channel['channel'].upper()}: Moderate diminishing returns (current ROI {returns_ratio:.1f}x higher than marginal)"
        else:
            analysis += f"\n- {channel['channel'].upper()}: Minimal diminishing returns (current ROI {returns_ratio:.1f}x higher than marginal)"
    
    if chart_description:
        analysis += f"\n\nMethodology: {chart_description}"
    
    return analysis


def get_roi_cpik_confidence_insights(soup):
    """Extract ROI and CPIK insights with confidence intervals"""
    import re
    import json
    
    # Find both charts
    roi_chart = soup.find("chart-embed", {"id": "roi-channel-chart"})
    cpik_chart = soup.find("chart-embed", {"id": "cpik-channel-chart"})
    
    if not roi_chart and not cpik_chart:
        return "No ROI or CPIK charts found."
    
    roi_data = []
    cpik_data = []
    cpik_description = None
    
    # Extract ROI data
    if roi_chart:
        roi_script = roi_chart.find_next("script", {"type": "text/javascript"})
        if roi_script:
            roi_data = extract_chart_data(roi_script.get_text())
    
    # Extract CPIK data and description
    if cpik_chart:
        cpik_element = cpik_chart.find_parent("chart")
        if cpik_element:
            cpik_desc = cpik_element.find("chart-description")
            cpik_description = cpik_desc.get_text(strip=True) if cpik_desc else None
        
        cpik_script = cpik_chart.find_next("script", {"type": "text/javascript"})
        if cpik_script:
            cpik_data = extract_chart_data(cpik_script.get_text())
    
    # Analyze the data
    analysis = analyze_roi_cpik_confidence(roi_data, cpik_data, cpik_description)
    
    return analysis

def extract_chart_data(script_content):
    """Extract data from chart script"""
    import re
    import json
    
    json_match = re.search(r'JSON\.parse\("(.+?)"\)', script_content, re.DOTALL)
    if json_match:
        try:
            escaped_json = json_match.group(1)
            unescaped_json = escaped_json.replace('\\"', '"').replace('\\n', '').replace('\\\\', '\\')
            chart_spec = json.loads(unescaped_json)
            
            datasets = chart_spec.get('datasets', {})
            for dataset_key, dataset_value in datasets.items():
                if isinstance(dataset_value, list):
                    return dataset_value
        except (json.JSONDecodeError, AttributeError):
            pass
    return []

def analyze_roi_cpik_confidence(roi_data, cpik_data, cpik_description):
    """Analyze ROI and CPIK data with confidence intervals"""
    
    analysis = "ROI and CPIK Performance Analysis with Confidence Intervals:\n"
    
    # ROI Analysis with Confidence Intervals
    if roi_data:
        analysis += f"\nROI Performance with 90% Credible Intervals:"
        
        roi_sorted = sorted(roi_data, key=lambda x: x['roi'], reverse=True)
        
        for i, channel in enumerate(roi_sorted, 1):
            roi_range = channel['ci_hi'] - channel['ci_lo']
            uncertainty = (roi_range / channel['roi']) * 100
            
            analysis += f"\n{i}. {channel['channel'].upper()}:"
            analysis += f"\n   - Point Estimate: {channel['roi']:.2f}x ROI"
            analysis += f"\n   - Confidence Range: {channel['ci_lo']:.2f}x to {channel['ci_hi']:.2f}x"
            analysis += f"\n   - Uncertainty Level: ±{uncertainty:.0f}% around point estimate"
        
        # ROI Confidence Analysis
        analysis += f"\n\nROI Confidence Analysis:"
        
        # Find most/least certain estimates
        roi_uncertainties = []
        for channel in roi_data:
            uncertainty = ((channel['ci_hi'] - channel['ci_lo']) / channel['roi']) * 100
            roi_uncertainties.append((channel['channel'], uncertainty, channel['ci_hi'] - channel['ci_lo']))
        
        most_certain = min(roi_uncertainties, key=lambda x: x[1])
        least_certain = max(roi_uncertainties, key=lambda x: x[1])
        
        analysis += f"\n- Most Reliable ROI Estimate: {most_certain[0].upper()} (±{most_certain[1]:.0f}% uncertainty)"
        analysis += f"\n- Least Reliable ROI Estimate: {least_certain[0].upper()} (±{least_certain[1]:.0f}% uncertainty)"
        
        # Overlapping confidence intervals analysis
        analysis += f"\n\nROI Statistical Significance:"
        for i, ch1 in enumerate(roi_data):
            for ch2 in roi_data[i+1:]:
                # Check if confidence intervals overlap
                if ch1['ci_lo'] <= ch2['ci_hi'] and ch2['ci_lo'] <= ch1['ci_hi']:
                    analysis += f"\n- {ch1['channel'].upper()} and {ch2['channel'].upper()}: Overlapping confidence intervals (no statistically significant difference)"
                else:
                    higher = ch1 if ch1['roi'] > ch2['roi'] else ch2
                    lower = ch2 if ch1['roi'] > ch2['roi'] else ch1
                    analysis += f"\n- {higher['channel'].upper()} significantly outperforms {lower['channel'].upper()} (non-overlapping intervals)"
    
    # CPIK Analysis with Confidence Intervals
    if cpik_data:
        analysis += f"\n\nCPIK Performance with 90% Credible Intervals:"
        
        cpik_sorted = sorted(cpik_data, key=lambda x: x['cpik'])  # Lower CPIK is better
        
        for i, channel in enumerate(cpik_sorted, 1):
            cpik_range = channel['ci_hi'] - channel['ci_lo']
            uncertainty = (cpik_range / channel['cpik']) * 100
            
            analysis += f"\n{i}. {channel['channel'].upper()} (Best to Worst CPIK):"
            analysis += f"\n   - Point Estimate: ${channel['cpik']:.3f} per KPI unit"
            analysis += f"\n   - Confidence Range: ${channel['ci_lo']:.3f} to ${channel['ci_hi']:.3f}"
            analysis += f"\n   - Uncertainty Level: ±{uncertainty:.0f}% around point estimate"
        
        # CPIK Confidence Analysis
        analysis += f"\n\nCPIK Confidence Analysis:"
        
        cpik_uncertainties = []
        for channel in cpik_data:
            uncertainty = ((channel['ci_hi'] - channel['ci_lo']) / channel['cpik']) * 100
            cpik_uncertainties.append((channel['channel'], uncertainty))
        
        most_certain_cpik = min(cpik_uncertainties, key=lambda x: x[1])
        least_certain_cpik = max(cpik_uncertainties, key=lambda x: x[1])
        
        analysis += f"\n- Most Reliable CPIK Estimate: {most_certain_cpik[0].upper()} (±{most_certain_cpik[1]:.0f}% uncertainty)"
        analysis += f"\n- Least Reliable CPIK Estimate: {least_certain_cpik[0].upper()} (±{least_certain_cpik[1]:.0f}% uncertainty)"
        
        # CPIK Statistical Significance
        analysis += f"\n\nCPIK Statistical Significance:"
        for i, ch1 in enumerate(cpik_data):
            for ch2 in cpik_data[i+1:]:
                if ch1['ci_lo'] <= ch2['ci_hi'] and ch2['ci_lo'] <= ch1['ci_hi']:
                    analysis += f"\n- {ch1['channel'].upper()} and {ch2['channel'].upper()}: Overlapping CPIK intervals (no statistically significant difference)"
                else:
                    better = ch1 if ch1['cpik'] < ch2['cpik'] else ch2  # Lower CPIK is better
                    worse = ch2 if ch1['cpik'] < ch2['cpik'] else ch1
                    analysis += f"\n- {better['channel'].upper()} significantly more cost-efficient than {worse['channel'].upper()}"
    
    # Combined ROI and CPIK Analysis
    if roi_data and cpik_data:
        analysis += f"\n\nCombined ROI and CPIK Insights:"
        
        # Match channels between datasets
        combined_data = []
        for roi_ch in roi_data:
            cpik_ch = next((c for c in cpik_data if c['channel'] == roi_ch['channel']), None)
            if cpik_ch:
                combined_data.append({
                    'channel': roi_ch['channel'],
                    'roi': roi_ch['roi'],
                    'roi_uncertainty': ((roi_ch['ci_hi'] - roi_ch['ci_lo']) / roi_ch['roi']) * 100,
                    'cpik': cpik_ch['cpik'],
                    'cpik_uncertainty': ((cpik_ch['ci_hi'] - cpik_ch['ci_lo']) / cpik_ch['cpik']) * 100
                })
        
        # Find channels with high confidence in both metrics
        high_confidence_channels = [
            ch for ch in combined_data 
            if ch['roi_uncertainty'] < 50 and ch['cpik_uncertainty'] < 50  # Less than 50% uncertainty
        ]
        
        if high_confidence_channels:
            analysis += f"\n\nHigh Confidence Performers (Low uncertainty in both ROI and CPIK):"
            for ch in sorted(high_confidence_channels, key=lambda x: x['roi'], reverse=True):
                analysis += f"\n- {ch['channel'].upper()}: {ch['roi']:.2f}x ROI (±{ch['roi_uncertainty']:.0f}%), ${ch['cpik']:.3f} CPIK (±{ch['cpik_uncertainty']:.0f}%)"
        
        # Uncertainty patterns
        analysis += f"\n\nUncertainty Patterns:"
        avg_roi_uncertainty = sum(ch['roi_uncertainty'] for ch in combined_data) / len(combined_data)
        avg_cpik_uncertainty = sum(ch['cpik_uncertainty'] for ch in combined_data) / len(combined_data)
        
        analysis += f"\n- Average ROI uncertainty: ±{avg_roi_uncertainty:.0f}%"
        analysis += f"\n- Average CPIK uncertainty: ±{avg_cpik_uncertainty:.0f}%"
        
        if avg_roi_uncertainty > avg_cpik_uncertainty:
            analysis += f"\n- ROI estimates show higher uncertainty than CPIK estimates"
        else:
            analysis += f"\n- CPIK estimates show higher uncertainty than ROI estimates"
    
    # Model Reliability Assessment
    analysis += f"\n\nModel Reliability Assessment:"
    
    if roi_data:
        narrow_roi_intervals = [ch for ch in roi_data if ((ch['ci_hi'] - ch['ci_lo']) / ch['roi']) < 0.3]
        wide_roi_intervals = [ch for ch in roi_data if ((ch['ci_hi'] - ch['ci_lo']) / ch['roi']) > 0.7]
        
        if narrow_roi_intervals:
            analysis += f"\n- High ROI Confidence: {', '.join([ch['channel'].upper() for ch in narrow_roi_intervals])} (narrow intervals)"
        if wide_roi_intervals:
            analysis += f"\n- Low ROI Confidence: {', '.join([ch['channel'].upper() for ch in wide_roi_intervals])} (wide intervals)"
    
    if cpik_data:
        narrow_cpik_intervals = [ch for ch in cpik_data if ((ch['ci_hi'] - ch['ci_lo']) / ch['cpik']) < 0.3]
        wide_cpik_intervals = [ch for ch in cpik_data if ((ch['ci_hi'] - ch['ci_lo']) / ch['cpik']) > 0.7]
        
        if narrow_cpik_intervals:
            analysis += f"\n- High CPIK Confidence: {', '.join([ch['channel'].upper() for ch in narrow_cpik_intervals])} (narrow intervals)"
        if wide_cpik_intervals:
            analysis += f"\n- Low CPIK Confidence: {', '.join([ch['channel'].upper() for ch in wide_cpik_intervals])} (wide intervals)"
    
    if cpik_description:
        analysis += f"\n\nMethodology Note: {cpik_description}"
    
    return analysis


def extract_response_curves_data_for_rag(soup):
    """Extract response curves data and format for RAG input with structured analysis"""
    import re
    import json
    from datetime import datetime
    
    # Find the response curves chart
    chart = soup.find("chart-embed", {"id": "response-curves-chart"})
    
    if not chart:
        return []
    
    # Extract script content
    script = chart.find_next("script", {"type": "text/javascript"})
    if not script:
        return []
    
    # Extract JSON data from script
    json_match = re.search(r'JSON\.parse\("(.+?)"\)', script.get_text(), re.DOTALL)
    if not json_match:
        return []
    
    try:
        # Clean and parse JSON
        escaped_json = json_match.group(1)
        unescaped_json = escaped_json.replace('\\"', '"').replace('\\n', '').replace('\\\\', '\\')
        chart_spec = json.loads(unescaped_json)
        
        # Get datasets
        datasets = chart_spec.get('datasets', {})
        raw_data = []
        for dataset_key, dataset_value in datasets.items():
            if isinstance(dataset_value, list):
                raw_data = dataset_value
                break
        
        if not raw_data:
            return []
        
        # Organize data by channel
        channels = {}
        for point in raw_data:
            channel = point['channel']
            if channel not in channels:
                channels[channel] = {
                    'name': channel,
                    'data_points': [],
                    'current_spend': None,
                    'current_revenue': None
                }
            
            channels[channel]['data_points'].append(point)
            
            if point.get('current_spend') == "Current spend":
                channels[channel]['current_spend'] = point['spend']
                channels[channel]['current_revenue'] = point['mean']
        
        # Calculate metrics for each channel
        channel_metrics = {}
        for channel_name, channel_data in channels.items():
            if not channel_data['current_spend']:
                continue
                
            sorted_points = sorted(channel_data['data_points'], key=lambda x: x['spend_multiplier'])
            current_roi = channel_data['current_revenue'] / channel_data['current_spend']
            
            # Calculate spend increase scenarios
            scenarios = {}
            target_multipliers = [1.25, 1.5, 2.0]
            
            for multiplier in target_multipliers:
                closest_point = min(sorted_points, key=lambda x: abs(x['spend_multiplier'] - multiplier))
                if closest_point['spend_multiplier'] >= multiplier:
                    additional_spend = closest_point['spend'] - channel_data['current_spend']
                    additional_revenue = closest_point['mean'] - channel_data['current_revenue']
                    marginal_roi = additional_revenue / additional_spend if additional_spend > 0 else 0
                    
                    scenarios[f"{int((multiplier-1)*100)}%"] = {
                        'additional_spend': additional_spend,
                        'additional_revenue': additional_revenue,
                        'marginal_roi': marginal_roi,
                        'new_total_spend': closest_point['spend'],
                        'new_total_revenue': closest_point['mean']
                    }
            
            # Calculate marginal returns at different levels
            marginal_returns = []
            for i in range(1, min(6, len(sorted_points))):
                if sorted_points[i]['spend_multiplier'] > 1.0:
                    prev_point = sorted_points[i-1]
                    curr_point = sorted_points[i]
                    spend_diff = curr_point['spend'] - prev_point['spend']
                    revenue_diff = curr_point['mean'] - prev_point['mean']
                    if spend_diff > 0:
                        marginal_roi = revenue_diff / spend_diff
                        marginal_returns.append({
                            'spend_multiplier': curr_point['spend_multiplier'],
                            'marginal_roi': marginal_roi
                        })
            
            # Find efficiency threshold
            efficiency_threshold = None
            for point in sorted_points:
                if point['spend_multiplier'] > 1.0:
                    point_roi = point['mean'] / point['spend'] if point['spend'] > 0 else 0
                    if point_roi < current_roi * 0.9:  # 10% drop threshold
                        efficiency_threshold = point['spend_multiplier']
                        break
            
            channel_metrics[channel_name] = {
                'current_spend': channel_data['current_spend'],
                'current_revenue': channel_data['current_revenue'],
                'current_roi': current_roi,
                'scenarios': scenarios,
                'marginal_returns': marginal_returns,
                'efficiency_threshold': efficiency_threshold,
                'max_modeled_spend': max(point['spend'] for point in sorted_points),
                'max_modeled_revenue': max(point['mean'] for point in sorted_points)
            }
        
        # Create structured analysis document
        timestamp = datetime.now().isoformat()
        
        # Calculate totals
        total_spend = sum(m['current_spend'] for m in channel_metrics.values())
        total_revenue = sum(m['current_revenue'] for m in channel_metrics.values())
        overall_roi = total_revenue / total_spend if total_spend > 0 else 0
        
        # Build analysis content
        content = "Marketing Response Curves Performance Analysis:\n\n"
        
        # Current Performance Summary
        content += "Current Channel Performance:\n"
        sorted_channels = sorted(channel_metrics.items(), key=lambda x: x[1]['current_roi'], reverse=True)
        
        for i, (channel, metrics) in enumerate(sorted_channels, 1):
            spend_pct = (metrics['current_spend'] / total_spend) * 100
            revenue_pct = (metrics['current_revenue'] / total_revenue) * 100
            content += f"{i}. {channel.upper()}:\n"
            content += f"   - Current Spend: ${metrics['current_spend']:,.0f} ({spend_pct:.1f}% of total)\n"
            content += f"   - Current Revenue: ${metrics['current_revenue']:,.0f} ({revenue_pct:.1f}% of total)\n"
            content += f"   - Current ROI: {metrics['current_roi']:.2f}x\n"
        
        content += f"\nPortfolio Summary:\n"
        content += f"- Total Spend: ${total_spend:,.0f}\n"
        content += f"- Total Revenue: ${total_revenue:,.0f}\n"
        content += f"- Overall ROI: {overall_roi:.2f}x\n"
        content += f"- Active Channels: {len(channel_metrics)}\n\n"
        
        # Spend Increase Scenarios
        content += "Spend Increase Scenario Analysis:\n\n"
        
        for scenario_pct in ["25%", "50%", "100%"]:
            content += f"{scenario_pct} Spend Increase Scenarios:\n"
            
            total_additional_spend = 0
            total_additional_revenue = 0
            scenario_count = 0
            
            for channel, metrics in sorted_channels:
                if scenario_pct in metrics['scenarios']:
                    scenario = metrics['scenarios'][scenario_pct]
                    content += f"- {channel.upper()}: +${scenario['additional_spend']:,.0f} → +${scenario['additional_revenue']:,.0f} (Marginal ROI: {scenario['marginal_roi']:.2f}x)\n"
                    total_additional_spend += scenario['additional_spend']
                    total_additional_revenue += scenario['additional_revenue']
                    scenario_count += 1
            
            if total_additional_spend > 0:
                portfolio_marginal_roi = total_additional_revenue / total_additional_spend
                content += f"Portfolio {scenario_pct} Increase: +${total_additional_spend:,.0f} → +${total_additional_revenue:,.0f} (Portfolio Marginal ROI: {portfolio_marginal_roi:.2f}x)\n\n"
        
        # Marginal Returns Analysis
        content += "Marginal Returns Analysis:\n"
        for channel, metrics in sorted_channels:
            if metrics['marginal_returns']:
                content += f"{channel.upper()} Marginal Returns:\n"
                for i, mr in enumerate(metrics['marginal_returns'][:3]):
                    content += f"   - At {mr['spend_multiplier']:.1f}x current spend: {mr['marginal_roi']:.2f}x marginal ROI\n"
        content += "\n"
        
        # Efficiency Threshold Analysis
        content += "Efficiency Threshold Analysis:\n"
        for channel, metrics in sorted_channels:
            if metrics['efficiency_threshold']:
                content += f"- {channel.upper()}: Efficiency maintained until {metrics['efficiency_threshold']:.1f}x current spend\n"
            else:
                content += f"- {channel.upper()}: Efficiency maintained beyond modeled range\n"
        content += "\n"
        
        # Scale and Opportunity Analysis
        content += "Scale and Growth Opportunity Analysis:\n"
        
        # Most efficient vs largest channels
        most_efficient = sorted_channels[0]
        largest_spend = max(channel_metrics.items(), key=lambda x: x[1]['current_spend'])
        largest_revenue = max(channel_metrics.items(), key=lambda x: x[1]['current_revenue'])
        
        content += f"- Most Efficient Channel: {most_efficient[0].upper()} ({most_efficient[1]['current_roi']:.2f}x ROI)\n"
        content += f"- Largest Spend Channel: {largest_spend[0].upper()} (${largest_spend[1]['current_spend']:,.0f})\n"
        content += f"- Highest Revenue Channel: {largest_revenue[0].upper()} (${largest_revenue[1]['current_revenue']:,.0f})\n"
        
        # Scale ratios
        spend_ratios = []
        for channel, metrics in channel_metrics.items():
            ratio = largest_spend[1]['current_spend'] / metrics['current_spend']
            spend_ratios.append((channel, ratio))
        
        spend_ratios.sort(key=lambda x: x[1], reverse=True)
        content += f"\nSpend Scale Differences:\n"
        for channel, ratio in spend_ratios:
            if ratio > 1:
                content += f"- {largest_spend[0].upper()} spends {ratio:.1f}x more than {channel.upper()}\n"
        
        content += "\n"
        
        # Portfolio Concentration Analysis
        spend_concentration = (largest_spend[1]['current_spend'] / total_spend) * 100
        revenue_concentration = (largest_revenue[1]['current_revenue'] / total_revenue) * 100
        
        content += "Portfolio Concentration:\n"
        content += f"- Spend Concentration: Top channel ({largest_spend[0].upper()}) represents {spend_concentration:.1f}% of total spend\n"
        content += f"- Revenue Concentration: Top channel ({largest_revenue[0].upper()}) represents {revenue_concentration:.1f}% of total revenue\n"
        
        high_share_channels = len([ch for ch, m in channel_metrics.items() if (m['current_spend']/total_spend) > 0.1])
        content += f"- Channels with >10% spend share: {high_share_channels} of {len(channel_metrics)}\n\n"
        
        # Growth Potential Summary
        content += "Growth Potential Summary:\n"
        for channel, metrics in sorted_channels:
            max_growth = (metrics['max_modeled_revenue'] / metrics['current_revenue']) - 1
            max_spend_multiplier = metrics['max_modeled_spend'] / metrics['current_spend']
            content += f"- {channel.upper()}: Up to {max_growth*100:.0f}% revenue growth potential (max {max_spend_multiplier:.1f}x current spend)\n"
        
        content += f"\nMethodology Note: Response curves show cumulative incremental revenue from total media spend over the selected time period, constructed based on historical flighting patterns."
        
        
        return content
        
    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        return []
    

result = ""

result = get_channel_contribution(soup) + "\n" + get_spend_outcome_insights(soup) + "\n" + get_channel_time_insights_with_anomalies(soup)['formatted_output'] + "\n" + get_roi_insights(soup) + "\n"+ get_roi_effectiveness_insights(soup) + "\n" + get_roi_marginal_insights(soup) + "\n" + get_roi_cpik_confidence_insights(soup) + extract_response_curves_data_for_rag(soup)

# Save the result to a text file
with open("summary_output/summary_extract_output.txt", "w") as f:
    f.write(result)

print("Result saved to output.txt")


