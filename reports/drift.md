# Drift monitoring and intercept adaptation

Oct 21–27 predictions are model-in-sample and Oct 28 is calibration-in-sample; they show error moving with the mix and are not held-out metrics. Labels for day d are assumed available at 00:00 on day d+1 because the data has no availability timestamps.

Unseen-C14 share is zero by construction during the Oct 21–27 refit window. The prototype makes one update on the partial Oct 30 day. Its monotonic intercept shift changes probability levels, never candidate order.

## Daily

| day | rows | clicks | ctr | site_side_share | unseen_c14_share | c20_sentinel_share | mean_pred | log_loss | partial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 141021 | 111065 | 19901 | 0.179183 | 0.626120 | 0.000000 | 0.511439 | 0.171519 | 0.400884 | False |
| 141022 | 143836 | 23967 | 0.166627 | 0.603368 | 0.000000 | 0.511840 | 0.161639 | 0.377784 | False |
| 141023 | 104452 | 20017 | 0.191638 | 0.684238 | 0.000000 | 0.491096 | 0.190205 | 0.422984 | False |
| 141024 | 89378 | 16475 | 0.184329 | 0.673555 | 0.000000 | 0.438945 | 0.184970 | 0.413812 | False |
| 141025 | 90218 | 17371 | 0.192545 | 0.685462 | 0.000000 | 0.428008 | 0.190435 | 0.420021 | False |
| 141026 | 103948 | 20267 | 0.194972 | 0.694001 | 0.000000 | 0.444424 | 0.191465 | 0.425299 | False |
| 141027 | 86788 | 16964 | 0.195465 | 0.657925 | 0.000000 | 0.426891 | 0.190881 | 0.418264 | False |
| 141028 | 142909 | 23631 | 0.165357 | 0.629946 | 0.342407 | 0.447900 | 0.168739 | 0.396069 | False |
| 141029 | 104450 | 18039 | 0.172705 | 0.633921 | 0.391996 | 0.455548 | 0.177329 | 0.410925 | False |
| 141030 | 22956 | 3800 | 0.165534 | 0.384780 | 0.594964 | 0.603328 | 0.169227 | 0.402740 | True |

## Hour-matched CTR

| period | rows | clicks | ctr |
| --- | --- | --- | --- |
| Oct21-28 h00-05 | 174422 | 32093 | 0.183996 |
| Oct29-30 h00-05 | 46863 | 7995 | 0.170604 |
| Oct21-28 all | 872594 | 158593 | 0.181749 |
| Oct29-30 all | 127406 | 21839 | 0.171413 |

## Population stability index

| column | day | psi |
| --- | --- | --- |
| app_category | 141029 | 0.031280 |
| app_category | 141030 | 0.369107 |
| site_category | 141029 | 0.037875 |
| site_category | 141030 | 0.376166 |
| device_type | 141029 | 0.004077 |
| device_type | 141030 | 0.003666 |
| C20 | 141029 | 0.151285 |
| C20 | 141030 | 0.536424 |

## Adaptation

| state | day | log_loss | mean_pred | mean_actual | rows | delta | shrink |
| --- | --- | --- | --- | --- | --- | --- | --- |
| frozen | 141029 | 0.410925 | 0.177329 | 0.172705 | 104450 | -0.017674 | 0.500000 |
| frozen | 141030 | 0.402740 | 0.169227 | 0.165534 | 22956 | -0.017674 | 0.500000 |
| adapted | 141030 | 0.402695 | 0.166981 | 0.165534 | 22956 | -0.017674 | 0.500000 |
