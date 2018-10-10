def precision(seg, gt):
    s_count = len(seg)
    g_count = len(gt)
    sp = 0
    gp = 0
    scp = 0
    gcp = 0
    right_count = 0
    while sp < s_count or gp < g_count:
        while scp != gcp:
            if scp < gcp:
                scp += len(seg[sp])
                sp += 1
            else:
                gcp += len(gt[gp])
                gp += 1

        if len(seg[sp]) == len(gt[gp]):
            right_count += 1

        scp += len(seg[sp])
        gcp += len(gt[gp])
        sp += 1
        gp += 1

    return right_count / s_count


def recall(seg, gt):
    s_count = len(seg)
    g_count = len(gt)
    sp = 0
    gp = 0
    scp = 0
    gcp = 0
    right_count = 0
    while sp < s_count or gp < g_count:
        while scp != gcp:
            if scp < gcp:
                scp += len(seg[sp])
                sp += 1
            else:
                gcp += len(gt[gp])
                gp += 1

        if len(seg[sp]) == len(gt[gp]):
            right_count += 1

        scp += len(seg[sp])
        gcp += len(gt[gp])
        sp += 1
        gp += 1

    return right_count / g_count


def f_measure(p, r):
    return 2 * p * r / (p + r)
