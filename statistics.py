from scipy import stats
import numpy as np

# Example paired t-test function
def paired_t(motion, no_motion):
    differences = motion - no_motion
    t_stat, p_val = stats.ttest_rel(motion, no_motion)
    cohens_d = np.mean(differences) / np.std(differences, ddof=1)
    return p_val, cohens_d

# Example Wilcoxon function
def wilcoxon(motion, no_motion):
    differences = motion - no_motion
    w_stat, p_val = stats.wilcoxon(motion, no_motion)
    r = w_stat / np.sqrt(len(differences))
    return p_val, r

def statistics(metric):
    """
    metric: numpy array of shape (6,6)
    Columns: [cond1_no, cond2_no, cond3_no, cond1_motion, cond2_motion, cond3_motion]
    """
    num_conditions = 3
    num_participants = metric.shape[0]

    results = [0] * num_conditions

    # Split no-motion and motion data per condition
    no_motion = metric[:, :num_conditions]  # shape (6,3)
    motion = metric[:, num_conditions:]  # shape (6,3)

    for k in range(num_conditions):
        # Extract values for this condition across participants
        no = no_motion[:, k]
        mo = motion[:, k]

        # Compute differences
        difference = mo - no

        # Normality test
        shapiro_test = stats.shapiro(difference)

        if shapiro_test.pvalue > 0.05:
            # Paired t-test
            p_val, effect_size = paired_t(mo, no)
        else:
            # Wilcoxon signed-rank test
            p_val, effect_size = wilcoxon(mo, no)

        # Store results
        results[k] = [float(p_val), float(effect_size)]

    return results