import pandas as pd

channel_list = {
    'Google': ['google_spend', 'google_impressions', 'google_clicks'],
    'Meta': ['meta_spend', 'meta_impressions', 'meta_clicks'],
    'TikTok': ['tiktok_spend', 'tiktok_impressions', 'tiktok_clicks'],
    'DV360': ['dv360_spend', 'dv360_impressions', 'dv360_clicks'],
    'X1': ['x1_spend', 'x1_impressions', 'x1_clicks'],
}

field_list = [
    ('google_spend', 'Google spend'),
    ('google_impressions', 'Google impressions'),
    ('google_clicks', 'Google clicks'),
    ('meta_spend', 'Meta spend'),
    ('meta_impressions', 'Meta impressions'),
    ('meta_clicks', 'Meta clicks'),
    ('meta_reach', 'Meta reach'),
    ('meta_frequency', 'Meta frequency'),
    ('tiktok_spend', 'Tiktok spend'),
    ('tiktok_impressions', 'Tiktok impressions'),
    ('tiktok_clicks', 'Tiktok clicks'),
    ('tiktok_conversion', 'Tiktok conversions'),
    ('tiktok_reach', 'Tiktok reach'),
    ('dv360_spend', 'DV360 spend'),
    ('dv360_impressions', 'DV360 impressions'),
    ('dv360_clicks', 'DV360 clicks'),
    ('x1_impressions', 'X1 impressions'),
    ('x1_clicks', 'X1 clicks'),
    ('x1_spend', 'X1 spend'),
    ('Quantity', 'Quantity sold'),
    ('dv_360_x1_impressions', 'DV360 X1 impressions'),
    ('dv_360_x1_clicks', 'DV360 X1 clicks'),
    ('dv_360_x1_spend', 'DV360 X1 spend'),
    ('Total Revenue', 'Total revenue'),
    ('total_discount', 'Total discount'),
    ('Service Fee', 'Service fee'),
    ('competitor_trend', 'Competitor trend'),
    ('CCI', 'CCI'),
    ('holidays', 'Holiday'),
    ('all_promo', 'Promotion'),
    ('seller_discount', 'Seller discount')
]

def row_to_sentences(row):
    date = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])
    sentence = f"On {date}, "
    details = []
    for col, desc in field_list:
        val = row[col]
        if col in ['holidays', 'all_promo']:
            if val == '1' or val == 1:
                details.append(f"{desc} is active")
            else:
                details.append(f"{desc} is not active")
        else:
            if pd.isna(val) or val == 0 or val == '0':
                details.append(f"{desc} is 0")
            else:
                details.append(f"{desc} is {val}")
    sentence += ', '.join(details) + '.'
    return sentence

def describe_change(curr, prev, metric_name, channel_name, is_currency=False):
    try:
        curr = float(curr)
        prev = float(prev)
    except:
        pass  # in case of non-numeric
    if pd.isna(curr) or pd.isna(prev):
        return f"{channel_name} {metric_name} data is missing."
    if curr == prev:
        if curr == 0:
            return f"{channel_name} {metric_name} remains at 0."
        else:
            return f"{channel_name} {metric_name} did not change and is {curr}."
    elif curr > prev:
        change = curr - prev
        if is_currency:
            return f"{channel_name} {metric_name} increased by ${change:,.2f} to ${curr:,.2f}."
        else:
            return f"{channel_name} {metric_name} increased by {change:,} to {curr:,}."
    else:
        change = prev - curr
        if is_currency:
            return f"{channel_name} {metric_name} decreased by ${change:,.2f} to ${curr:,.2f}."
        else:
            return f"{channel_name} {metric_name} decreased by {change:,} to {curr:,}."

def generate_change_sentences(df):
    """
    Given a DataFrame with a 'date' column and the channel metrics,
    generate a list of change sentences (week-over-week) for each metric in each channel.
    """
    df = df.sort_values('date').reset_index(drop=True)
    currency_metrics = ['google_spend', 'meta_spend', 'tiktok_spend', 'dv360_spend', 'x1_spend']
    change_sentences = []
    for i in range(1, len(df)):
        curr_row = df.iloc[i]
        prev_row = df.iloc[i-1]
        date = curr_row['date'].strftime('%Y-%m-%d') if hasattr(curr_row['date'], 'strftime') else str(curr_row['date'])
        for channel, metrics in channel_list.items():
            for metric in metrics:
                curr_val = curr_row[metric]
                prev_val = prev_row[metric]
                is_currency = metric in currency_metrics
                sentence = describe_change(curr_val, prev_val, metric.replace('_', ' '), channel, is_currency)
                change_sentences.append(f"On {date}, {sentence}")
    return change_sentences

if __name__ == '__main__':
    # Parse dates to ensure proper sorting and formatting
    df = pd.read_csv('dataset/Hitachi_dataset - FULL_merged_output (5).csv', parse_dates=['date'])
    sentences = df.apply(row_to_sentences, axis=1)

    with open('rag_sentences.txt', 'w', encoding='utf-8') as f:
        for s in sentences:
            f.write(s + '\n')
    print("Conversion complete. Sentences saved to rag_sentences.txt")

    # Generate and save change sentences
    change_sentences = generate_change_sentences(df)
    with open('rag_change_sentences.txt', 'w', encoding='utf-8') as f:
        for s in change_sentences:
            f.write(s + '\n')
    print("Change sentences saved to rag_change_sentences.txt")