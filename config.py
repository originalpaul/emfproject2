START_DATE = "2016-01-01"
END_DATE   = "2025-09-01"            
BENCH_TICKER = "MCHI"        # msci china thing

# yfinance uses .HK suffix for hnog kogn
UNIVERSE = [
    "0700.HK",  # Tencent
    "9988.HK",  # Alibaba HK
    "3690.HK",  # Meituan
    "9618.HK",  # JD.com HK
    "1024.HK",  # Kuaishou
    "1211.HK",  # BYD
    "2318.HK",  # Ping An
    "0939.HK",  # CCB
    "1398.HK",  # ICBC
    "0981.HK",  # SMIC
    "0883.HK",  # CNOOC
    "0386.HK",  # Sinopec
    "0857.HK",  # PetroChina
    "2269.HK",  # Wuxi Biologics
    "BABA", "JD", "PDD", "BIDU", "NIO", "LI", "XPEV"
]

# portfolio settings
REB_FREQ = "W-FRI"   # rebalance every Friday close
TOP_K = 10           # hold top 10 names by model signal
HOLDING_PERIOD_DAYS = 5  # predict next 5 trading days return

# risk management (simple)
MAX_WEIGHT = 0.15            # per-name cap
CASH_BUFFER = 0.05           # keep 5% in cash
VOL_TARGET_ANNUAL = None     # set like 0.18 to scale exposure to target vol, or None
