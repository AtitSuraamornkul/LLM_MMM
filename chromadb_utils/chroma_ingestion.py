import os
import time
import re
import json
import uuid

import chromadb
from chromadb.config import Settings

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


persist_directory = "./chroma_db"
client = chromadb.PersistentClient(path=persist_directory)

# Or for in-memory storage (data will be lost when script ends)
# client = chromadb.Client()

# Collection name (equivalent to Pinecone index)
collection_name = "m150-thb"

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5" 
)

# Initialize Chroma vector store
vector_store = Chroma(
    client=client,
    collection_name=collection_name,
    embedding_function=embeddings,
    persist_directory=persist_directory  # Remove this line if using in-memory client
)

documents = [
    Document(
        page_content="Marketing Mix Model Optimization Results (Jul 4, 2022 - Jul 1, 2024): Budget optimization maintained THB45M spend while improving ROI from 3.2 to 3.4 (+0.1 delta) and incremental revenue from THB146M to THB152M (+THB6M delta). Channel-level spend constraints were -30% to +30% of historical spend. Budget allocation shifts: activation increased from 29% to 37%, tv_spot decreased from 26% to 18%, kol increased from 11% to 13%, youtube increased from 10% to 13%, facebook decreased from 16% to 12%, radio decreased from 7% to 6%, tv_sponsor maintained at 1%, kol_boost maintained at 1%, tiktok maintained at 0%.",
        metadata={"source": "optimization_report", "category": "optimization_summary", "time_period": "2022-07-04_to_2024-07-01"}
    ),
    Document(
        page_content="Optimized Spend Changes by Channel (THB): tv_spot decreased by THB-3,460,000, facebook decreased by THB-1,770,000, radio decreased by THB-570,000, tiktok decreased by THB-30,000, activation increased by THB3,800,000, youtube increased by THB1,310,000, kol increased by THB540,000, tv_sponsor increased by THB100,000, kol_boost increased by THB80,000. Optimized budget allocation (THB): activation THB16,760,000 (37.1%), tv_spot THB8,080,000 (17.9%), youtube THB5,680,000 (12.6%), kol THB5,680,000 (12.6%), facebook THB5,380,000 (11.9%), radio THB2,730,000 (6.0%), tv_sponsor THB440,000 (1.0%), kol_boost THB350,000 (0.8%), tiktok THB110,000 (0.2%).",
        metadata={"source": "optimization_report", "category": "spend_allocation", "currency": "THB"}
    ),
    Document(
        page_content="Optimized Incremental Revenue Impact by Channel (THB): Non-optimized baseline revenue was THB145,745,072. Channel revenue changes after optimization: activation increased by THB7,363,328, youtube increased by THB2,974,818, kol increased by THB977,014, tv_sponsor increased by THB273,048.62, kol_boost increased by THB179,869.88, facebook decreased by THB-2,683,926, tv_spot decreased by THB-2,371,982, radio decreased by THB-897,651, tiktok decreased by THB-47,465.62. Total optimization impact: THB5,767,053.88 increase.",
        metadata={"source": "optimization_report", "category": "revenue_optimization", "currency": "THB"}
    ),
    Document(
        page_content="Channel Response Curve Analysis: ACTIVATION - No clear diminishing return point found. Optimal spend range THB9.07M to THB16.85M. At maximum spend, revenue uncertainty ranges from THB31.72M to THB115.40M. Optimized spend THB16,760,000 vs non-optimized THB12,960,000. TV_SPOT - Optimal spend range THB8.08M to THB15.00M. Revenue uncertainty THB10.97M to THB30.06M. Optimized spend THB8,080,000 vs non-optimized THB11,540,000. KOL - Optimal spend range THB3.60M to THB6.68M. Revenue uncertainty THB17.96M to THB37.96M. Optimized spend THB5,680,000 vs non-optimized THB5,140,000.",
        metadata={"source": "optimization_report", "category": "response_curves", "analysis_type": "spend_optimization"}
    ),
    Document(
        page_content="Channel Response Curve Analysis Continued: YOUTUBE - No clear diminishing return point found. Optimal spend range THB3.06M to THB5.68M. Revenue uncertainty THB18.02M to THB34.75M. Optimized spend THB5,680,000 vs non-optimized THB4,370,000. FACEBOOK - Optimal spend range THB5.00M to THB9.29M. Revenue uncertainty THB20.40M to THB47.39M. Optimized spend THB5,380,000 vs non-optimized THB7,150,000. RADIO - Optimal spend range THB2.31M to THB4.29M. Revenue uncertainty THB14.15M to THB22.68M. Optimized spend THB2,730,000 vs non-optimized THB3,300,000.",
        metadata={"source": "optimization_report", "category": "response_curves", "analysis_type": "spend_optimization"}
    ),
    Document(
        page_content="Channel Contribution Analysis: Total revenue THB1,209,910,157 split between baseline revenue 88.0% (THB1,064,185,216) and marketing channels 12.0% (THB145,724,941). Marketing channel performance by revenue contribution: ACTIVATION 4.1% (THB49,697,312), FACEBOOK 2.1% (THB25,648,736), KOL 1.6% (THB19,866,100), YOUTUBE 1.4% (THB17,402,620), RADIO 1.2% (THB14,916,628), TV_SPOT 1.2% (THB14,483,313), TV_SPONSOR 0.1% (THB1,706,650), KOL_BOOST 0.1% (THB1,326,770), TIKTOK 0.1% (THB676,812). ACTIVATION is the top performing marketing channel at 4.1% of total revenue.",
        metadata={"source": "summary_report", "category": "channel_contribution", "revenue_split": "baseline_vs_marketing"}
    ),
    Document(
        page_content="Channel ROI and Efficiency Analysis: ROI rankings (within media channels only): KOL_BOOST 5.0x ROI (0.9% revenue share, 0.6% spend share), TIKTOK 5.0x ROI (0.5% revenue share, 0.3% spend share), TV_SPONSOR 5.0x ROI (1.2% revenue share, 0.8% spend share), RADIO 4.5x ROI (10.2% revenue share, 7.3% spend share), YOUTUBE 4.0x ROI (11.9% revenue share, 9.7% spend share), KOL 3.9x ROI (13.6% revenue share, 11.4% spend share), ACTIVATION 3.8x ROI (34.1% revenue share, 28.7% spend share), FACEBOOK 3.6x ROI (17.6% revenue share, 15.8% spend share), TV_SPOT 1.3x ROI (9.9% revenue share, 25.5% spend share). Most efficient: KOL_BOOST (revenue/spend ratio 1.55). Least efficient: TV_SPOT (revenue/spend ratio 0.39).",
        metadata={"source": "summary_report", "category": "roi_analysis", "metric_type": "efficiency_ranking"}
    ),
    Document(
        page_content="Monthly Channel Performance with Anomalies: ACTIVATION - Average monthly revenue THB1,308,661, peak 2023-12 (THB1,376,579), active 9/24 months. FACEBOOK - Average THB291,767, peak 2022-11 (THB598,305), active 20/24 months, high growth periods 2023-06 (+668%), 2023-11 (+1767%), 2024-04 (+24442%), decline periods 2023-05 (-91%), 2023-07 (-89%), 2023-10 (-89%). KOL - Average THB438,111, peak 2024-04 (THB673,887), active 10/24 months, decline periods 2023-05 (-63%), 2024-01 (-69%), 2024-06 (-71%). KOL_BOOST - Average THB31,406, peak 2024-04 (THB69,155), active 9/24 months, high growth 2023-12 (+936%), decline periods 2023-05 (-81%), 2024-01 (-82%), 2024-05 (-72%).",
        metadata={"source": "summary_report", "category": "monthly_performance", "analysis_type": "anomaly_detection"}
    ),
    Document(
        page_content="Monthly Channel Performance Continued: RADIO - Average monthly revenue THB462,222, peak 2023-03 (THB890,620), active 7/24 months, decline periods 2022-11 (-64%), 2022-12 (-78%), 2023-05 (-83%). TIKTOK - Average THB23,522, peak 2023-02 (THB63,130), active 6/24 months, revenue spike 2023-02 (+168% above average), decline periods 2023-03 (-57%), 2023-04 (-64%), 2024-05 (-78%). TV_SPONSOR - Average THB115,942, peak 2024-04 (THB248,299), active 3/24 months, decline periods 2024-05 (-66%), 2024-06 (-81%). TV_SPOT - Average THB842,441, peak 2023-02 (THB1,349,152), active 4/24 months, decline 2023-05 (-86%). YOUTUBE - Average THB489,576, peak 2023-03 (THB849,421), active 8/24 months, decline 2024-06 (-76%).",
        metadata={"source": "summary_report", "category": "monthly_performance", "analysis_type": "anomaly_detection"}
    ),
    Document(
        page_content="Quarterly Channel Trends: ACTIVATION - Overall growth trend, best Q4-2023 (THB1,376,579), active 3 quarters, recent 2Q average THB1,376,579. FACEBOOK - Overall growth, best Q4-2022 (THB597,634), active 7 quarters, major growth Q2-2024, major decline Q2-2023 and Q1-2024, recent 2Q average THB124,782. KOL - Overall growth, best Q1-2023 (THB666,441), active 3 quarters, major decline Q2-2023 and Q1-2024, recent 2Q average THB64,354. KOL_BOOST - Overall growth, best Q1-2023 (THB50,203), active 3 quarters, major decline Q2-2023 and Q1-2024, recent 2Q average THB1,950. RADIO - Stable trend, best Q1-2023 (THB890,620), active 2 quarters, major growth Q1-2023, major decline Q2-2023, recent 2Q average THB0.",
        metadata={"source": "summary_report", "category": "quarterly_trends", "analysis_type": "growth_patterns"}
    ),
    Document(
        page_content="Quarterly Trends Continued: TIKTOK - Overall growth, best Q1-2023 (THB27,169), active 2 quarters, major decline Q2-2023, recent 2Q average THB634. TV_SPONSOR - Overall growth, best Q2-2024 (THB15,895), active 1 quarter, recent 2Q average THB7,948. TV_SPOT - Stable trend, best Q1-2023 (THB1,200,092), active 1 quarter, major decline Q2-2023, recent 2Q average THB0. YOUTUBE - Overall growth, best Q1-2023 (THB849,421), active 3 quarters, major decline Q2-2023 and Q4-2023, recent 2Q average THB73,422. Revenue anomaly analysis shows TIKTOK had highest spike in 2023-02 (+168% above average). Most stable channel: ACTIVATION (7% volatility). Most volatile: TV_SPONSOR (103% volatility).",
        metadata={"source": "summary_report", "category": "quarterly_trends", "analysis_type": "volatility_analysis"}
    ),
    Document(
        page_content="Channel Performance Rankings: Total Revenue Rankings: 1. ACTIVATION THB11,777,945, 2. FACEBOOK THB5,835,347, 3. KOL THB4,381,111, 4. YOUTUBE THB3,916,611, 5. TV_SPOT THB3,369,766, 6. RADIO THB3,235,556, 7. TV_SPONSOR THB347,826, 8. KOL_BOOST THB282,656, 9. TIKTOK THB141,133. Consistency Rankings (% months active): 1. FACEBOOK 83%, 2. KOL 42%, 3. ACTIVATION 38%, 4. KOL_BOOST 38%, 5. YOUTUBE 33%, 6. RADIO 29%, 7. TIKTOK 25%, 8. TV_SPOT 17%, 9. TV_SPONSOR 12%. Peak Performance Rankings: 1. ACTIVATION THB1,376,579, 2. TV_SPOT THB1,349,152, 3. RADIO THB890,620, 4. YOUTUBE THB849,421, 5. KOL THB673,887, 6. FACEBOOK THB598,305, 7. TV_SPONSOR THB248,299, 8. KOL_BOOST THB69,155, 9. TIKTOK THB63,130.",
        metadata={"source": "summary_report", "category": "performance_rankings", "ranking_type": "comprehensive"}
    ),
    Document(
        page_content="Channel Lifecycle Analysis: KOL - Active/Mature (2023-02 to 2024-06), TV_SPOT - Dormant (2023-02 to 2023-05), TV_SPONSOR - Launch Phase (2024-04 to 2024-06), RADIO - Dormant (2022-10 to 2023-05), ACTIVATION - Active/Mature (2023-10 to 2024-06), FACEBOOK - Active/Mature (2022-10 to 2024-06), YOUTUBE - Active/Mature (2023-02 to 2024-06), TIKTOK - Active/Mature (2023-02 to 2024-06), KOL_BOOST - Active/Mature (2023-03 to 2024-06). 6-Month Momentum Analysis: KOL strong positive momentum (+809%), TV_SPOT neutral (+0%), TV_SPONSOR neutral (+0%), RADIO neutral (+0%), ACTIVATION neutral (+0%), FACEBOOK strong positive (+571%), YOUTUBE neutral (+0%), TIKTOK neutral (+0%), KOL_BOOST strong positive (+928%).",
        metadata={"source": "summary_report", "category": "lifecycle_analysis", "analysis_type": "momentum_tracking"}
    ),
    Document(
        page_content="6-Month Momentum Detailed Analysis: KOL recent 3-month average THB413,395 vs previous 3-month average THB45,479 (20% months showed growth). FACEBOOK recent average THB312,366 vs previous THB46,584 (20% growth months). KOL_BOOST recent average THB30,768 vs previous THB2,992 (20% growth months). TV_SPONSOR recent average THB115,942 vs previous THB0 (20% growth months). ACTIVATION recent average THB1,322,824 vs previous THB1,322,824 (80% growth months). All other channels showed neutral momentum with 0% or 20% growth consistency.",
        metadata={"source": "summary_report", "category": "momentum_details", "analysis_type": "trend_consistency"}
    ),
    Document(
        page_content="ROI vs Effectiveness Analysis: Average effectiveness 4.0057 incremental outcome per impression, total media spend THB45,197,221. ROI Rankings: 1. KOL_BOOST 5.0x, 2. TIKTOK 5.0x, 3. TV_SPONSOR 5.0x, 4. RADIO 4.5x, 5. YOUTUBE 4.0x, 6. KOL 3.9x, 7. ACTIVATION 3.8x, 8. FACEBOOK 3.6x, 9. TV_SPOT 1.3x. Effectiveness Rankings: 1. KOL_BOOST 5.0005, 2. TIKTOK 5.0002, 3. TV_SPONSOR 4.9981, 4. RADIO 4.5270, 5. YOUTUBE 3.9836, 6. KOL 3.8645, 7. ACTIVATION 3.8338, 8. FACEBOOK 3.5885, 9. TV_SPOT 1.2550. Star Performers (high ROI and effectiveness): TV_SPONSOR, RADIO, TIKTOK, KOL_BOOST. Optimization Needed: KOL, TV_SPOT, ACTIVATION, FACEBOOK, YOUTUBE.",
        metadata={"source": "summary_report", "category": "roi_effectiveness", "metric_type": "performance_matrix"}
    ),
    Document(
        page_content="Detailed Spend Allocation by Channel (THB): 1. ACTIVATION THB12,963,048 (28.7%), 2. TV_SPOT THB11,540,248 (25.5%), 3. FACEBOOK THB7,147,552 (15.8%), 4. KOL THB5,140,650 (11.4%), 5. YOUTUBE THB4,368,572 (9.7%), 6. RADIO THB3,295,000 (7.3%), 7. TV_SPONSOR THB341,463 (0.8%), 8. KOL_BOOST THB265,330 (0.6%), 9. TIKTOK THB135,358 (0.3%). Strategic insights: KOL_BOOST delivers highest ROI (5.0x) with lowest spend allocation, TV_SPONSOR shows ideal performance with high ROI and effectiveness, ACTIVATION receives largest budget but shows below-average efficiency requiring optimization.",
        metadata={"source": "summary_report", "category": "spend_allocation_detailed", "analysis_type": "budget_distribution"}
    ),
    Document(
        page_content="Marginal ROI and Saturation Analysis: Average ROI 4.0x, average Marginal ROI 1.9x. Marginal ROI Rankings: 1. TV_SPONSOR 3.0x, 2. KOL_BOOST 2.5x, 3. YOUTUBE 2.5x, 4. ACTIVATION 2.2x, 5. KOL 1.9x, 6. TIKTOK 1.5x, 7. RADIO 1.4x, 8. FACEBOOK 1.3x, 9. TV_SPOT 0.6x. Saturation Indicators: KOL moderate saturation (ROI/Marginal ROI ratio 2.1x), TV_SPOT moderate saturation (2.1x), TV_SPONSOR low saturation (1.7x), RADIO high saturation (3.2x), ACTIVATION low saturation (1.8x), FACEBOOK high saturation (2.7x), YOUTUBE low saturation (1.6x), TIKTOK high saturation (3.4x), KOL_BOOST low saturation (2.0x). Diminishing Returns: Strong patterns in RADIO and TIKTOK, minimal in TV_SPONSOR, ACTIVATION, YOUTUBE, KOL_BOOST.",
        metadata={"source": "summary_report", "category": "marginal_roi", "analysis_type": "saturation_assessment"}
    ),
    Document(
        page_content="Efficiency Gap Analysis: KOL 2.0x gap between current and marginal returns, TV_SPOT 0.7x gap, TV_SPONSOR 2.0x gap, RADIO 3.1x gap, ACTIVATION 1.7x gap, FACEBOOK 2.3x gap, YOUTUBE 1.5x gap, TIKTOK 3.5x gap, KOL_BOOST 2.5x gap. Performance consistency shows ROI variation 1.11 standard deviation, Marginal ROI variation 0.70 standard deviation. Channels with low spend allocation but above-average marginal efficiency: KOL (11.4% spend), TV_SPONSOR (0.8% spend), YOUTUBE (9.7% spend), KOL_BOOST (0.6% spend).",
        metadata={"source": "summary_report", "category": "efficiency_gaps", "analysis_type": "performance_consistency"}
    ),
    Document(
        page_content="ROI and CPIK Confidence Intervals (90% Credible): ROI with uncertainty levels: KOL_BOOST 5.00x (±4%), TIKTOK 5.00x (±2%), TV_SPONSOR 5.00x (±5%), RADIO 4.53x (±45%), YOUTUBE 3.98x (±57%), KOL 3.86x (±69%), ACTIVATION 3.83x (±115%), FACEBOOK 3.59x (±86%), TV_SPOT 1.26x (±95%). CPIK (cost per incremental KPI): KOL_BOOST THB0.200 (±4%), TIKTOK THB0.200 (±2%), TV_SPONSOR THB0.200 (±5%), RADIO THB0.223 (±45%), YOUTUBE THB0.254 (±58%), KOL THB0.264 (±71%), ACTIVATION THB0.269 (±142%), FACEBOOK THB0.287 (±89%), TV_SPOT THB0.827 (±103%). Most reliable estimates: TIKTOK (±2% uncertainty). Least reliable: ACTIVATION (±115% ROI, ±142% CPIK uncertainty).",
        metadata={"source": "summary_report", "category": "confidence_intervals", "metric_type": "statistical_significance"}
    ),
    Document(
        page_content="Detailed ROI Confidence Ranges: KOL_BOOST 4.91x to 5.10x, TIKTOK 4.95x to 5.05x, TV_SPONSOR 4.88x to 5.12x, RADIO 3.58x to 5.61x, YOUTUBE 2.95x to 5.22x, KOL 2.66x to 5.33x, ACTIVATION 1.84x to 6.24x, FACEBOOK 2.25x to 5.32x, TV_SPOT 0.73x to 1.92x. CPIK Confidence Ranges: KOL_BOOST THB0.196 to THB0.204, TIKTOK THB0.198 to THB0.202, TV_SPONSOR THB0.195 to THB0.205, RADIO THB0.178 to THB0.279, YOUTUBE THB0.192 to THB0.339, KOL THB0.188 to THB0.376, ACTIVATION THB0.160 to THB0.543, FACEBOOK THB0.188 to THB0.445, TV_SPOT THB0.520 to THB1.374.",
        metadata={"source": "summary_report", "category": "detailed_confidence_ranges", "metric_type": "interval_bounds"}
    ),
    Document(
        page_content="Statistical Significance Analysis: Channels with statistically significant ROI differences (non-overlapping confidence intervals): FACEBOOK, KOL, KOL_BOOST, RADIO, TIKTOK, TV_SPONSOR, YOUTUBE all significantly outperform TV_SPOT. All other channel pairs show overlapping confidence intervals indicating no statistically significant ROI differences. CPIK significance: FACEBOOK, KOL, KOL_BOOST, RADIO, TIKTOK, TV_SPONSOR, YOUTUBE all significantly more cost-efficient than TV_SPOT. Model reliability assessment shows high confidence for KOL_BOOST, TIKTOK, TV_SPONSOR (narrow intervals) and low confidence for ACTIVATION, FACEBOOK, TV_SPOT (wide intervals). Average ROI uncertainty ±53%, average CPIK uncertainty ±58%. CPIK estimates show higher uncertainty than ROI estimates.",
        metadata={"source": "summary_report", "category": "statistical_significance", "analysis_type": "confidence_comparison"}
    ),
    Document(
        page_content="Marketing Response Curves Current Performance (Top 7 Channels): 1. TV_SPONSOR - Current spend THB341,463 (0.8%), revenue THB1,706,650 (1.2%), ROI 5.00x. 2. RADIO - Current spend THB3,295,000 (7.4%), revenue THB14,916,628 (10.4%), ROI 4.53x. 3. YOUTUBE - Current spend THB4,368,572 (9.8%), revenue THB17,402,620 (12.1%), ROI 3.98x. 4. KOL - Current spend THB5,140,650 (11.5%), revenue THB19,866,100 (13.8%), ROI 3.86x. 5. ACTIVATION - Current spend THB12,963,048 (28.9%), revenue THB49,697,312 (34.6%), ROI 3.83x. 6. FACEBOOK - Current spend THB7,147,552 (16.0%), revenue THB25,648,736 (17.8%), ROI 3.59x. 7. TV_SPOT - Current spend THB11,540,248 (25.8%), revenue THB14,483,313 (10.1%), ROI 1.26x. Portfolio total spend THB44,796,533, overall ROI 3.21x.",
        metadata={"source": "summary_report", "category": "response_curves", "analysis_type": "current_performance"}
    ),
    Document(
        page_content="Detailed Spend Increase Scenario Analysis - 25% Increase: TV_SPONSOR +THB85,366 spend → +THB235,487 revenue (2.76x marginal ROI), RADIO +THB823,750 → +THB1,048,923 (1.27x), YOUTUBE +THB1,092,143 → +THB2,514,652 (2.30x), KOL +THB1,285,162 → +THB2,200,658 (1.71x), ACTIVATION +THB3,240,762 → +THB6,376,468 (1.97x), FACEBOOK +THB1,786,888 → +THB2,126,128 (1.19x), TV_SPOT +THB2,885,062 → +THB1,542,903 (0.53x). Portfolio 25% increase: +THB11,199,133 spend → +THB16,045,219 revenue (1.43x portfolio marginal ROI).",
        metadata={"source": "summary_report", "category": "scenario_analysis_25", "analysis_type": "spend_scaling"}
    ),
    Document(
        page_content="Detailed Spend Increase Scenario Analysis - 50% and 100%: 50% Increase - TV_SPONSOR +THB170,732 → +THB438,110 (2.57x), RADIO +THB1,647,500 → +THB1,895,140 (1.15x), YOUTUBE +THB2,184,286 → +THB4,709,210 (2.16x), KOL +THB2,570,325 → +THB4,038,400 (1.57x), ACTIVATION +THB6,481,524 → +THB11,700,384 (1.81x), FACEBOOK +THB3,573,776 → +THB3,862,276 (1.08x), TV_SPOT +THB5,770,124 → +THB2,819,871 (0.49x). Portfolio 50% increase marginal ROI: 1.32x. 100% Increase - Portfolio marginal ROI: 1.14x. TV_SPONSOR maintains 2.26x marginal ROI, YOUTUBE 1.92x, ACTIVATION 1.55x, while RADIO drops to 0.97x and FACEBOOK to 0.92x.",
        metadata={"source": "summary_report", "category": "scenario_analysis_50_100", "analysis_type": "spend_scaling"}
    ),
    Document(
        page_content="Growth Potential and Portfolio Concentration: Revenue growth potential at maximum efficient spend: TV_SPONSOR 52%, YOUTUBE 55%, ACTIVATION 46%, KOL 40%, RADIO 24%, FACEBOOK 29%, TV_SPOT 38%. Spend scale differences: ACTIVATION spends 38.0x more than TV_SPONSOR, 3.9x more than RADIO, 3.0x more than YOUTUBE, 2.5x more than KOL, 1.8x more than FACEBOOK, 1.1x more than TV_SPOT. Portfolio concentration: Top channel (ACTIVATION) represents 28.9% of total spend and 34.6% of total revenue. 4 of 7 channels have >10% spend share. Efficiency thresholds: TV_SPONSOR, YOUTUBE, ACTIVATION maintain efficiency until 1.3x current spend; RADIO, KOL, FACEBOOK, TV_SPOT until 1.2x current spend.",
        metadata={"source": "summary_report", "category": "growth_potential", "analysis_type": "portfolio_optimization"}
    ),
    Document(
        page_content="Key Business Insights and Methodology Notes: Business context shows ACTIVATION and FACEBOOK drove most overall revenue. Baseline revenue accounts for 88.0% of total revenue while marketing channels drive 12.0%. Combined total revenue THB1,209,910,157 (Baseline THB1,064,185,216 + Marketing THB145,724,941). ROI insights: For every THB1 spent on KOL_BOOST, THB5.00 in revenue was generated. KOL_BOOST had highest effectiveness and drove lowest CPIK at THB0.20. TV_SPONSOR had highest marginal ROI at 2.98x. Methodology: ROI calculated by dividing revenue attributed to channel by marketing costs. Effectiveness measures incremental outcome per impression. Response curves show cumulative incremental revenue from total media spend constructed based on historical flighting patterns. CPIK point estimate determined by posterior median, ROI by posterior mean.",
        metadata={"source": "combined", "category": "business_insights", "analysis_type": "methodology_summary"}
    )
]



# 1. Get all IDs
all_ids = vector_store.get()['ids']

# 2. Delete by IDs (if any exist)
if all_ids:
    vector_store.delete(ids=all_ids)

print("deleted")


# Generate unique IDs
uuids = [str(uuid.uuid4()) for _ in documents]

# Add documents to ChromaDB
vector_store.add_documents(documents=documents, ids=uuids)

print(f"Successfully added {len(documents)} documents to ChromaDB collection '{collection_name}'")

all_data = vector_store.get()
print(f"📊 Total documents in database: {len(all_data['ids'])}")