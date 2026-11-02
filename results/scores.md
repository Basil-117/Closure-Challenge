| Model | **Overall** | alpha_15_13929_4048 | alpha_15_13929_2024 | alpha_05_4071_4048 | alpha_05_4071_2024 | AR_1_Ret_360 | AR_3_Ret_360 | AR_14_Ret_180 | NASA_2DWMH |
|---|---|---|---|---|---|---|---|---|---|
| k-ω SST baseline (no ML) | **0.1036** | 0.1320 | 0.2049 | 0.0461 | 0.0719 | 0.1288 | 0.1243 | 0.0590 | 0.0621 |
| LightGBM | **0.0656** | 0.0322 | 0.0602 | 0.0505 | 0.0770 | 0.0883 | 0.0839 | 0.0373 | 0.0959 |
| LightGBM + SST fallback | **0.0632** | 0.0322 | 0.0602 | 0.0505 | 0.0770 | 0.0883 | 0.0839 | 0.0373 | 0.0761 |

| Rank | Entry | Overall score |
|---|---|---|
| 1 | Yang | 0.0580 |
| 2 | Reissmann, Fang, Sandberg | 0.0595 |
| 3 | Wu and Zhang | 0.0624 |
| 4 | **LightGBM + SST fallback (this repo)** | 0.0632 |
| 5 | Tian, Buchanan, Hickel, Dwight | 0.0641 |
| 6 | **LightGBM (this repo)** | 0.0656 |
| 7 | Liu, Wang, Zhao, Xiao | 0.0737 |
| 8 | Montoya, Oulghelou, Cinnella | 0.0779 |
| 9 | **k-ω SST baseline (no ML) (this repo)** | 0.1036 |
