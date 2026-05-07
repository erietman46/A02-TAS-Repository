import numpy as np
import scipy.stats as stats
import pandas as pd

condition_name = {
    1: "Fixed-base Position",
    2: "Fixed-base Velocity",
    3: "Fixed-base Acceleration",
    4: "Motion-base Position",
    5: "Motion-base Velocity",
    6: "Motion-base Acceleration",
}

dist_gain_crossover = {
    1: np.array([1.9754735318170151, 1.7009249743460206, 1.41076920037148,   2.740800048368519,  2.045174000628029,  np.nan]),
    2: np.array([2.6467233075674126, 1.7946737158482764, 2.5358460161659293, 2.038925425079407,  1.9311249916189783, 2.2796750810935578]),
    3: np.array([2.9748902193741538, 1.7952998982101094, 2.1214691868080635, 2.2153220314928475, 2.4775828788903986, 2.181180175293617]),
    4: np.array([2.2648539807932986, 1.8927479568626089, 1.9159901077059318, 2.439587249185493,  2.5800569853510926, 1.4641811853635016]),
    5: np.array([2.7195007153636035, 1.9937585994953269, 2.0286466872287883, 2.889769488893627,  2.8454472993737907, 2.324719049755511]),
    6: np.array([3.9130150220476465, 2.922891786330172,  2.3369321616766983, 2.702353198656887,  3.303601863102024,  2.96736952611436]),
}

dist_phase_margin = {
    1: np.array([85.3981896118395, 107.62530663500175, 90.44649198022796, 57.65673369893729, 86.0905402497752,  np.nan]),
    2: np.array([53.6348831094048, 62.71398538926992, 48.33223303707018, 64.314301037086,   72.04892208840809, 65.38889740593375]),
    3: np.array([24.781446436619802, 33.816303718759514, 19.200099085091324, 20.06620432574283, 29.09240661762965, 22.16193241227458]),
    4: np.array([86.45786897893484, 101.07331284594032, 79.68533573069855, 58.666517004535706, 69.13699474302103, 104.26611250357554]),
    5: np.array([43.71319982806091, 71.56426441593251, 51.457875463731085, 39.595086276686175, 51.50744361526105, 66.37439595824624]),
    6: np.array([19.516565880782792, 25.494448251563313, 31.532684120549277, 22.999596813499465, 20.61572410704457, 23.13553055760434]),
}

tgt_gain_crossover = {
    1: np.array([1.9754735318170151, 1.7009249743460206, 1.41076920037148,   2.740800048368519,  2.045174000628029,  np.nan]),
    2: np.array([2.6467233075674126, 1.7946737158482764, 2.5358460161659293, 2.038925425079407,  1.9311249916189783, 2.2796750810935578]),
    3: np.array([2.9748902193741538, 1.7952998982101094, 2.1214691868080635, 2.2153220314928475, 2.4775828788903986, 2.181180175293617]),
    4: np.array([1.7900005090151054, 1.1633660175696436, 1.675886180257961,  1.9008663278480262, 2.562886774990454,  1.182668375645899]),
    5: np.array([2.637079633217062,  1.1655170673139563, 1.6842266605871592, 1.981417927256623,  1.8959283095796284, 1.458704217559575]),
    6: np.array([3.2356651190839116, 1.9211841819090096, 1.920025108620212,  1.8986310266805342, 2.257744011677711,  1.8267099997691902]),
}

tgt_phase_margin = {
    1: np.array([85.3981896118395, 107.62530663500175, 90.44649198022796, 57.65673369893729, 86.0905402497752,  np.nan]),
    2: np.array([53.6348831094048, 62.71398538926992, 48.33223303707018, 64.314301037086,   72.04892208840809, 65.38889740593375]),
    3: np.array([24.781446436619802, 33.816303718759514, 19.200099085091324, 20.06620432574283, 29.09240661762965, 22.16193241227458]),
    4: np.array([84.48370682068273, 103.40736273665286, 79.78903499425263, 70.54109912537405, 53.33677334027311, 91.69227472504178]),
    5: np.array([48.55297526496946, 73.50780742979623, 59.860455767860444, 57.817026911098424, 63.11217729956573, 69.83664815224265]),
    6: np.array([33.66193972285984, 51.56669372114206, 35.67173269194737, 52.40012863987752, 43.95012446737451, 52.66675922890093]),
}

#welch test definition
def welch_ttest(a, b):
    #ommits nan values
    return stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")

def ttest_row(metric_block, metric_name, cond_a, cond_b, a, b):
    res = welch_ttest(a, b)
    return {
        "metric_block": metric_block,    
        "metric": metric_name,            
        "cond_a": cond_a,
        "cond_b": cond_b,
        "cond_a_name": condition_name[cond_a],
        "cond_b_name": condition_name[cond_b],
        "n_a": int(np.sum(~np.isnan(a))),
        "n_b": int(np.sum(~np.isnan(b))),
        "mean_a": float(np.nanmean(a)),
        "mean_b": float(np.nanmean(b)),
        "t_stat": float(res.statistic),
        "p_value": float(res.pvalue),
    }

pairs = [
    ("position", 1, 4),
    ("velocity", 2, 5),
    ("acceleration", 3, 6),
]

results = []

for label, cA, cB in pairs:
    print(f"\n=== {label.upper()} (c{cA} vs c{cB}) ===")

    # Disturbance
    r = ttest_row("disturbance", "gain_crossover_rad_s", cA, cB,
                  dist_gain_crossover[cA], dist_gain_crossover[cB])
    results.append(r)
    print(f"Dist gain crossover: t={r['t_stat']:.6g}, p={r['p_value']:.6g}")

    r = ttest_row("disturbance", "phase_margin_deg", cA, cB,
                  dist_phase_margin[cA], dist_phase_margin[cB])
    results.append(r)
    print(f"Dist phase margin:   t={r['t_stat']:.6g}, p={r['p_value']:.6g}")

    # Target
    r = ttest_row("target", "gain_crossover_rad_s", cA, cB,
                  tgt_gain_crossover[cA], tgt_gain_crossover[cB])
    results.append(r)
    print(f"Tgt gain crossover:  t={r['t_stat']:.6g}, p={r['p_value']:.6g}")

    r = ttest_row("target", "phase_margin_deg", cA, cB,
                  tgt_phase_margin[cA], tgt_phase_margin[cB])
    results.append(r)
    print(f"Tgt phase margin:    t={r['t_stat']:.6g}, p={r['p_value']:.6g}")

# Save results
df = pd.DataFrame(results)
df.to_csv("t_test_results_nonparametric.csv", index=False)
print("\nSaved: t_test_results_nonparametric.csv")