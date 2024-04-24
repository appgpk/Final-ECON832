import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import skew

def CPC18_getDist(H, pH, L, lot_shape, lot_num):
    if lot_shape == '-':
        if pH == 1:
            dist = np.array([H, pH])
            dist.shape = (1, 2)
        else:
            dist = np.array([[L, 1-pH], [H, pH]])

    else:  # H is multi outcome
        # compute H distribution
        high_dist = np.zeros(shape=(lot_num, 2))
        if lot_shape == 'Symm':
            k = lot_num - 1
            for i in range(0, lot_num):
                high_dist[i, 0] = H - k / 2 + i
                high_dist[i, 1] = pH * stats.binom.pmf(i, k, 0.5)

        elif (lot_shape == 'R-skew') or (lot_shape == 'L-skew'):
            if lot_shape == 'R-skew':
                c = -1 - lot_num
                dist_sign = 1
            else:
                c = 1 + lot_num
                dist_sign = -1
            for i in range(1, lot_num+1):
                high_dist[i - 1, 0] = H + c + dist_sign * pow(2, i)
                high_dist[i - 1, 1] = pH / pow(2, i)

            high_dist[lot_num - 1, 1] = high_dist[lot_num - 1, 1] * 2

        # incorporate L into the distribution
        dist = np.copy(high_dist)
        locb = np.where(high_dist[:, 0] == float(L))
        if locb[0].size > 0:
            dist[locb, 1] += (1-pH)
        elif pH < 1:
            dist = np.vstack((dist, [L, 1-pH]))

        dist = dist[np.argsort(dist[:, 0])]

    return dist

def CPC15_isStochasticDom (DistA, DistB):
    na = DistA.shape[0]
    nb = DistB.shape[0]
    if np.array_equal(DistA, DistB):
        dom = False
        which = None
    else:
        tempa = np.ones(shape=(na, 1))
        tempb = np.ones(shape=(nb, 1))
        for i in range(0, nb):
            sumpa = 0
            j = 0
            sumpb = np.sum(DistB[0:i + 1, 1])

            while (sumpa != 1) and (j < na) and (sumpa + DistA[j, 1] <= sumpb):
                sumpa += DistA[j, 1]
                if sumpa == sumpb:
                    break
                j += 1

            if j == na:
                j = na - 1
            if i == nb:
                i = nb - 1

            if DistB[i, 0] < DistA[j, 0]:
                tempb[i] = 0
                break

        if np.all(tempb != 0):
            dom = True
            which = 'B'
        else:
            for i in range(0, na):
                sumpb = 0
                j = 0
                sumpa = np.sum(DistA[0: i+1, 1])

                while (sumpb != 1) and (j < nb) and (sumpb + DistB[j, 1] <= sumpa):
                    sumpb += DistB[j, 1]
                    if sumpa == sumpb:
                        break
                    j += 1

                if j == nb:
                    j = nb - 1
                if i == na:
                    i = na - 1

                if DistA[i, 0] < DistB[j, 0]:
                    tempa[i] = 0
                    break

            if np.all(tempa != 0):
                dom = True
                which = 'A'
            else:
                dom = False
                which = None

    return pd.DataFrame([{'dom': dom, 'which': which}])


data = pd.read_csv("raw-comp-set-data-Track-2.csv", delimiter=";")

H = data["Ha"]
pH = data["pHa"]
L = data["La"]
lot_shape = data["LotShapeA"].astype(str)
lot_num = data["LotNumA"]
resultA = [CPC18_getDist(H[i], pH[i], L[i], lot_shape[i], lot_num[i]) for i in range(len(H))]

H = data["Hb"]
pH = data["pHb"]
L = data["Lb"]
lot_shape = data["LotShapeB"].astype(str)
lot_num = data["LotNumB"]
resultB = [CPC18_getDist(H[i], pH[i], L[i], lot_shape[i], lot_num[i]) for i in range(len(H))]

######################################## Covariates for Risk Preference ################################################

varA = []
skewA = []
varB = []
skewB = []
EV_A = []
EV_B = []
for dist in resultA:
    values = dist[:, 0]
    probabilities = dist[:, 1]
    mean = np.sum(values * probabilities)
    variance = np.sum(probabilities * (values - mean) ** 2)
    skewness = 0 if variance == 0 else np.sum(probabilities * ((values - mean) / np.sqrt(variance)) ** 3)
    expected_value = np.sum(values * probabilities)
    EV_A.append(expected_value)
    varA.append(variance)
    skewA.append(skewness)

for dist in resultB:
    values = dist[:, 0]
    probabilities = dist[:, 1]
    mean = np.sum(values * probabilities)
    variance = np.sum(probabilities * (values - mean) ** 2)
    skewness = 0 if variance == 0 else np.sum(probabilities * ((values - mean) / np.sqrt(variance)) ** 3)
    expected_value = np.sum(values * probabilities)
    EV_B.append(expected_value)
    varB.append(variance)
    skewB.append(skewness)

Dominance = []

for i in range(len(H)):
    comparison_result = CPC15_isStochasticDom(resultA[i], resultB[i])
    dom_value = comparison_result['dom'].iloc[0]
    which_value = str(comparison_result['which'].iloc[0])
    if dom_value == False and which_value == 'None':
        Dominance.append(1)
    elif dom_value == True and which_value == 'A':
        Dominance.append(2)
    elif dom_value == True and which_value == 'B':
        Dominance.append(3)

HAPWA = []
LAPWA = []
for dist in resultA:
    values = dist[:, 0]
    probabilities = dist[:, 1]

    max_index = np.argmax(values)
    min_index = np.argmin(values)
    max_probability = probabilities[max_index]
    min_probability = probabilities[min_index]

    distorted_max_probability = max_probability *2
    distorted_min_probability = min_probability *0.5

    HAPWA.append(distorted_max_probability)
    LAPWA.append(distorted_min_probability)



HAPWB = []
LAPWB = []
for dist in resultB:
    values = dist[:, 0]
    probabilities = dist[:, 1]

    max_index = np.argmax(values)
    min_index = np.argmin(values)
    max_probability = probabilities[max_index]
    min_probability = probabilities[min_index]

    distorted_max_probability = max_probability *2
    distorted_min_probability = min_probability *0.5

    HAPWB.append(distorted_max_probability)
    LAPWB.append(distorted_min_probability)



############################### Covariates Relevant to Choice Context Focused on Attention #############################
order = data["Order"]
RelativeOrder = np.array(order) / 30.0


varA_df = pd.DataFrame({'varA': varA})
varB_df = pd.DataFrame({'varB': varB})
EV_A_df = pd.DataFrame({'EVA': EV_A})
EV_B_df = pd.DataFrame({'EVB': EV_B})
skewA_df = pd.DataFrame({'skewA': skewA})
skewB_df = pd.DataFrame({'skewB': skewB})
Dominance_df = pd.DataFrame({'Dominance': Dominance})
HAPWA_df = pd.DataFrame({'HAPWA': HAPWA})
LAPWA_df = pd.DataFrame({'LAPWA': LAPWA})
HAPWB_df = pd.DataFrame({'HAPWB': HAPWB})
LAPWB_df = pd.DataFrame({'LAPWB': LAPWB})
RelativeOrder_df = pd.DataFrame({'RelativeOrder': RelativeOrder})

data_combined = pd.concat([data, varA_df, varB_df, EV_A_df, EV_B_df, skewA_df, skewB_df,
                           Dominance_df, HAPWA_df, LAPWA_df, HAPWB_df, LAPWB_df, RelativeOrder_df], axis=1)

data_combined.to_csv('DataToExploit.csv', index=False)