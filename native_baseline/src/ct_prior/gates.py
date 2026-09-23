import math


def safety(candidate, reference):
    failures=[]
    if candidate["aggregation"]!=reference["aggregation"] or candidate["samples"]!=reference["samples"]:
        return ["evaluation protocol mismatch"]
    if set(candidate["patients"])!=set(reference["patients"]): return ["patient mismatch"]
    for patient, c in candidate["patients"].items():
        r=reference["patients"][patient]
        for key, limit, sign in (("psnr_db",.02,-1),("mae_hu",.05,1),("high_gradient_mae_hu",.05,1)):
            a,b=c.get(key),r.get(key)
            if a is None or b is None or not math.isfinite(a+b) or sign*(a-b)>limit:
                failures.append(patient+":"+key)
    return failures


def qualify(candidate, reference, minimum_gain=.1):
    failures=safety(candidate,reference)
    gain=reference["metrics"]["mae_hu"]-candidate["metrics"]["mae_hu"]
    if not math.isfinite(gain) or gain<minimum_gain: failures.append("MAE improvement below gate")
    return {"passed":not failures,"mae_gain_hu":gain,"failures":failures}


def selection_key(result):
    m=result["metrics"]
    return (m["psnr_db"],m["ssim"],-m["mae_hu"])
