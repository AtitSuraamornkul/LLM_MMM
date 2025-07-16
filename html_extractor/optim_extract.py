from bs4 import BeautifulSoup
import json


with open('output/new_optimization_output.html', 'r', encoding='utf-8') as f:
    soup = BeautifulSoup(f, 'html.parser')


def time_extractor(soup):
  time_period = ""
  time_chip = soup.find("chip")
  time_period += time_chip.text.strip()

  return time_period

    
def extract_scenario_summary(soup):
    scenario_card = soup.find("card", {"id": "scenario-plan"})
    stats = scenario_card.find_all("stats")
    optimization_scenario = ""

    scenario_summary = {}
    for stat in stats:
        title = stat.find("stats-title").text.strip()
        value = stat.find("stat").text.strip()
        delta_tag = stat.find("delta")
        delta = delta_tag.text.strip() if delta_tag else None
        scenario_summary[title] = {"value": value, "delta": delta}

    print("Optimization Scenario Summary:")
    for k, v in scenario_summary.items():
        optimization_scenario += f"- {k}: {v['value']}" + (f" (delta: {v['delta']})" if v['delta'] else "") + "\n"
        
    return optimization_scenario



def allocation_extractor(soup):
    budget_allocation = ""
    table_card = soup.find("chart-table", {"id": "spend-allocation-table"})
    rows = table_card.find_all("tr")[1:]  # skip headers

    allocation_data = []
    for row in rows:
        cols = row.find_all("td")
        allocation_data.append({
            "channel": cols[0].text.strip(),
            "non_optimized": cols[1].text.strip(),
            "optimized": cols[2].text.strip()
        })
    for row in allocation_data:
        budget_allocation += f"- {row['channel']}: {row['non_optimized']} to {row['optimized']}" + "\n"


    return "Budget Allocation Table:" + "\n" + budget_allocation


def insights_extractor(soup):
    paragraphs = [p.text.strip() for p in soup.find_all("p", class_="insights-text")]
    insights = ""
    for i, text in enumerate(paragraphs, 1):
        insights += f"{i}. {text}" + "\n"
    return "Insights:" + "\n" + insights


def spend_delta_chart_extractor(soup):
    chart_div = soup.find("chart-embed", {"id": "spend-delta-chart"})
    if not chart_div:
        return "Chart embed with id 'spend-delta-chart' not found."

    vega_script = chart_div.find_next("script", {"type": "text/javascript"})
    if not vega_script:
        return "No script tag found after chart embed."

    script_text = vega_script.text

    json_start = script_text.find('JSON.parse("') + len('JSON.parse("')
    json_end = script_text.find('")', json_start)
    json_str = script_text[json_start:json_end]

    clean_json = json_str.encode().decode('unicode_escape')
    spec = json.loads(clean_json)

    # Try to get the title
    title = spec.get('title')

    if isinstance(title, dict):
        chart_title = title.get('text', '')
    else:
        chart_title = title or ''


    chart_desc_tag = chart_div.find_next("chart-description")
    if chart_desc_tag:
        chart_description = chart_desc_tag.get_text(strip=True)

    # Extract dataset
    dataset_key = next(iter(spec['datasets']))
    chart_data = spec['datasets'][dataset_key]

    output =  f"Chart Title: {chart_title}" + "\n" + f"Description: {chart_description}" + "\n" + "Data Points:" + "\n" 

    for item in chart_data:
        change_type = "Increase" if item['spend'] > 0 else "Decrease"
        output += f"{item['channel']: <12}  Spend Change: ({change_type}) THB{item['spend']:,.0f}" + "\n"

    
    return output


def spend_allocation_chart_extractor(soup):
    chart_div = soup.find("chart-embed", id="spend-allocation-chart")
    if not chart_div:
        return "Chart embed with id 'spend-allocation-chart' not found."
    
    vega_script = chart_div.find_next("script", {"type": "text/javascript"})
    if not vega_script:
        return "No script tag found after chart embed."
    
    script_text = vega_script.text

    json_start = script_text.find('JSON.parse("') + len('JSON.parse("')
    json_end = script_text.find('")', json_start)
    json_str = script_text[json_start:json_end]

    cleaned_json = json_str.encode().decode('unicode_escape')
    spec = json.loads(cleaned_json)

    title = spec.get('title')
    if isinstance(title, dict):
        chart_title = title.get('text', '')
    else:
        chart_title = title or ''

    chart_desc_tag = chart_div.find_next("chart-description")
    if chart_desc_tag:
        chart_description = chart_desc_tag.get_text(strip=True)


    datasets = spec.get('datasets', {})
    if not datasets:
        raise ValueError("No datasets found in spec!")
    
    # Extract dataset
    dataset_key = next(iter(spec['datasets']))
    chart_data = spec['datasets'][dataset_key]

    output =  f"Chart Title: {chart_title}" + "\n" + f"Description: {chart_description}" + "\n" + "Data Points:" + "\n" 


    for item in chart_data:
        output += f"{item['channel']: <12} THB{item['spend']:,}" + "\n"

    #Calculate percentages
    total = sum(item['spend'] for item in chart_data)
    output += "\nPercentage Allocation:" + "\n"
    for item in chart_data:
        percentage = (item['spend'] / total) * 100
        output += f"{item['channel']: <12} {percentage:.1f}%" + "\n"

    return output


def outcome_delta_extractor(soup):
    chart_div = soup.find("chart-embed", {"id": "outcome-delta-chart"})
    if not chart_div:
        return "Chart embed with id 'outcome-delta-chart' not found."
    vega_script = chart_div.find_next("script", {"type": "text/javascript"})
    if not vega_script:
        return "No script tag found after chart embed."
    

    script_text = vega_script.text

    json_start = script_text.find('JSON.parse("') + len('JSON.parse("')
    json_end = script_text.find('")', json_start)
    json_str = script_text[json_start:json_end]

    cleaned_json = json_str.encode().decode('unicode_escape')
    spec = json.loads(cleaned_json)

    title = spec.get('title')
    if isinstance(title, dict):
        chart_title = title.get('text', '')
    else:
        chart_title = title or ''

    chart_desc_tag = chart_div.find_next("chart-description")
    if chart_desc_tag:
        chart_description = chart_desc_tag.get_text(strip=True)


    datasets = spec.get('datasets', {})
    if not datasets:
        raise ValueError("No datasets found in spec!")
    
    # Extract dataset
    dataset_key = next(iter(spec['datasets']))
    chart_data = spec['datasets'][dataset_key]

    # Find key data points
    non_optimized = next((x for x in chart_data if x['channel'] == 'non_optimized'), None)
    optimized = next((x for x in chart_data if x['channel'] == 'optimized'), None)
    channels = [x for x in chart_data if x['channel'] not in ['non_optimized', 'optimized']]
    
    # Calculate impacts
    total_impact = sum(x['incremental_outcome'] for x in channels)
    net_change = optimized['incremental_outcome'] - non_optimized['incremental_outcome'] if optimized and non_optimized else 0

    output =  f"Chart Title: {chart_title}" + "\n" + f"Description: {chart_description}" + "\n" + "\n"
    output += f"Non-optimized Revenue: THB{non_optimized['incremental_outcome']:,.2f}" + "\n" + "\n"
    output += "Channel revenue change:" + "\n"
    
    for item in sorted(channels, key=lambda x: abs(x['incremental_outcome']), reverse=True):
        change = "Increase" if item['incremental_outcome'] > 0 else "Decrease"
        output += f"{item['channel']: <12} THB{item['incremental_outcome']:>12,.2f} ({change})" + "\n"
    
    output += "\n" + f"Total Optimization Impact change: THB{total_impact:,.2f}" + "\n"
    output += f"Net Change: {'+' if net_change >= 0 else ''}{net_change:,.2f}" + "\n"
    output += f"Optimized Revenue change: THB{optimized['incremental_outcome']:,.2f}"

    return output


def extract_insights_from_html(soup):
    import json

    chart_div = soup.find("card", {"id": "optimized-response-curves"})
    vega_script = chart_div.find_next("script", {"type": "text/javascript"})
    if not vega_script:
        return "No script tag found after chart embed."
    
    json_text = vega_script.text

    json_start = json_text.find('JSON.parse("') + len('JSON.parse("')
    json_end = json_text.find('")', json_start)
    json_str = json_text[json_start:json_end]

    clean_json = json_str.encode().decode('unicode_escape')

    spec = json.loads(clean_json)
    dataset_key = next(iter(spec["datasets"]))
    data = spec["datasets"][dataset_key]

    # Organize data by channel
    channel_data = {}
    for d in data:
        channel = d["channel"].replace("_x1", "").lower()
        channel_data.setdefault(channel, []).append(d)

    summaries = []

    for channel, entries in channel_data.items():
        entries = [e for e in entries if e.get("spend") is not None and e.get("mean") is not None]
        entries.sort(key=lambda x: x["spend"])
        if not entries:
            continue

        max_entry = max(entries, key=lambda x: x["mean"])
        max_revenue = max_entry["mean"]
        ci_lo = max_entry.get("ci_lo")
        ci_hi = max_entry.get("ci_hi")

        optimized = next((e["spend"] for e in entries if str(e.get("spend_level", "")).lower() == "optimized spend"), None)
        non_optimized = next((e["spend"] for e in entries if str(e.get("spend_level", "")).lower() == "non-optimized spend"), None)

        max_spend = 1.3 * non_optimized if non_optimized else None
        min_spend = 0.7 * non_optimized if non_optimized else None

        spend_threshold = None
        rev_at_threshold = None

        if non_optimized:
            lower_limit = 0.7 * non_optimized
            upper_limit = 1.3 * non_optimized
            for e in entries:
                if e["spend"] < lower_limit or e["spend"] > upper_limit:
                    spend_threshold = e["spend"]
                    rev_at_threshold = e["mean"]
                    break

        # Compose a natural language paragraph
        summary = [f"Channel: {channel.capitalize()}"]

        if spend_threshold:
            summary.append(
                f"Spend beyond ±30% of THB{non_optimized:,.0f} (≈THB{spend_threshold:,.0f}) is considered to have diminishing returns."
            )
        else:
            summary.append("No clear diminishing return point was found.")

        if min_spend and max_spend:
            summary.append(
                f"The optimal spend range is estimated to be THB{min_spend/1e6:.2f}M to THB{max_spend/1e6:.2f}"
            )

        if ci_lo and ci_hi:
            summary.append(
                f"At the maximum spend, the expected revenue uncertainty ranges from THB{ci_lo/1e6:.2f}M to THB{ci_hi/1e6:.2f}M."
            )
        else:
            summary.append("Confidence interval at maximum spend is unavailable.")

        if optimized:
            summary.append(f"The optimized spend is THB{optimized:,.0f}.")
        if non_optimized:
            summary.append(f"The non-optimized spend is THB{non_optimized:,.0f}.")

        summaries.append("\n".join(summary))

    return "\n\n".join(summaries)


llm_input = ""
llm_input += time_extractor(soup) + "\n" + extract_scenario_summary(soup) + "\n" + allocation_extractor(soup) + "\n" + insights_extractor(soup) + "\n" + spend_delta_chart_extractor(soup) + "\n" + spend_allocation_chart_extractor(soup) + "\n" + outcome_delta_extractor(soup) + "\n" + extract_insights_from_html(soup)

print(llm_input)

with open('llm_input/llm_input.txt', 'w') as f:
    f.write(llm_input)

print("LLM input saved to llm_input.txt")



