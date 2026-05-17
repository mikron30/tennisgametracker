# Regression Anchor Events Active Ranges

Generated from the current accepted regression logs:
- `tmp/regression_0_8010_after_frame9889_fix_narrow.txt`
- `tmp/regression_9600_10100_after_frame9889_fix_narrow.txt`

Use these active windows for future comparisons:
- `0-8010`
- `9600-10100`

Skip `8011-9599` for now. The user-confirmed rest interval is `8010-9000`, and active comparison resumes from `9600`.

| seq | segment | frame | category | event | pos | vel | detail |
| ---: | --- | ---: | --- | --- | --- | ---: | --- |
| 1 | 0-8010 | 35 | rally | serve_start | `(2616,887)` |  | low=(2584, 1009) apex=(2616, 887) rise=122px up_steps=3 |
| 2 | 0-8010 | 65 | rally | point_end | `(1722,431)` | 9.5 | reason=Ball hit the net duration=30f vel_hist=[38.3, 32.8, 30.1, 26.2, 9.5] |
| 3 | 0-8010 | 148 | rally | serve_start | `(2515,925)` |  | low=(2489, 1027) apex=(2515, 925) rise=102px up_steps=3 |
| 4 | 0-8010 | 244 | rally | point_end | `(1428,558)` | 0.0 | reason=Ball bounced twice on court duration=96f vel_hist=[] |
| 5 | 0-8010 | 521 | rally | serve_start | `(1239,846)` |  | low=(1242, 986) apex=(1239, 846) rise=140px up_steps=3 |
| 6 | 0-8010 | 649 | rally | point_end | `` |  | reason=STUCK_TIMEOUT stuck=15 duration=128f |
| 7 | 0-8010 | 1177 | rally | serve_start | `(2613,880)` |  | low=(2580, 1010) apex=(2613, 880) rise=130px up_steps=3 |
| 8 | 0-8010 | 1327 | rally | point_end | `(1114,846)` | 37.5 | reason=Ball bounced out of court (left sideline) duration=150f vel_hist=[68.8, 72.7, 74.3, 63.5, 37.5] |
| 9 | 0-8010 | 1564 | rally | serve_start | `(1224,888)` |  | low=(1236, 1028) apex=(1224, 888) rise=140px up_steps=3 |
| 10 | 0-8010 | 1712 | rally | point_end | `(2218,190)` | 2.2 | reason=Ball hit upper fence and fell down duration=148f vel_hist=[15.1, 14.3, 11.2, 4.5, 2.2] |
| 11 | 0-8010 | 2065 | rally | serve_start | `(2623,886)` |  | low=(2591, 1007) apex=(2623, 886) rise=121px up_steps=3 |
| 12 | 0-8010 | 2104 | rally | point_end | `(1780,311)` | 20.9 | reason=Serve bounce outside left service box duration=39f vel_hist=[13.4, 12.2, 13.5, 11.4, 20.9] |
| 13 | 0-8010 | 2220 | rally | serve_start | `(2544,906)` |  | low=(2521, 1001) apex=(2544, 906) rise=95px up_steps=3 |
| 14 | 0-8010 | 2356 | rally | point_end | `(1464,432)` | 0.0 | reason=Ball hit the net duration=136f vel_hist=[28.2, 89.5, 29.0, 0.0, 0.0] |
| 15 | 0-8010 | 2805 | rally | serve_start | `(1184,885)` |  | low=(1191, 1021) apex=(1184, 885) rise=136px up_steps=3 |
| 16 | 0-8010 | 2904 | offscreen | upper_side_exit | `(1530,23)` |  | DEBUG: Ball likely exited through upper side from (1530,23) |
| 17 | 0-8010 | 2904 | offscreen | top_return_wait_activated | `(1530,23)` |  | DEBUG: [TOP-RETURN WAIT] activated for delayed upper-side re-entry search |
| 18 | 0-8010 | 2936 | onscreen | top_return_continuation | `(2056,41)` |  | DEBUG: [TOP-RETURN GRACE] prioritizing continuation candidate at (2056, 41) area=8.5px score=-7.0 motion=28.0/209.0 source=alt |
| 19 | 0-8010 | 2937 | onscreen | top_return_continuation | `(2061,59)` |  | DEBUG: [TOP-RETURN GRACE] prioritizing continuation candidate at (2061, 59) area=15.5px score=5.3 motion=14.5/163.0 source=alt |
| 20 | 0-8010 | 2938 | onscreen | top_return_continuation | `(2066,77)` |  | DEBUG: [TOP-RETURN GRACE] prioritizing continuation candidate at (2066, 77) area=26.0px score=134.2 motion=16.5/145.0 source=primary |
| 21 | 0-8010 | 2963 | rally | point_end | `` |  | reason=STUCK_TIMEOUT stuck=15 duration=158f |
| 22 | 0-8010 | 3263 | rally | serve_start | `(2618,900)` |  | low=(2585, 1024) apex=(2618, 900) rise=124px up_steps=3 |
| 23 | 0-8010 | 3295 | rally | point_end | `(1818,457)` | 23.0 | reason=Ball hit the net duration=32f vel_hist=[33.5, 29.6, 26.1, 13.0, 23.0] |
| 24 | 0-8010 | 3359 | rally | serve_start | `(2361,918)` |  | low=(2337, 1013) apex=(2361, 918) rise=95px up_steps=3 |
| 25 | 0-8010 | 3484 | rally | point_end | `(2068,526)` | 61.6 | reason=Ball bounced on hitter side after crossing net duration=125f vel_hist=[32.3, 30.4, 32.6, 32.8, 61.6] |
| 26 | 0-8010 | 3878 | rally | serve_start | `(1170,893)` |  | low=(1186, 997) apex=(1170, 893) rise=104px up_steps=3 |
| 27 | 0-8010 | 3909 | rally | point_end | `(2374,368)` | 29.5 | reason=Serve bounce outside right service box duration=31f vel_hist=[19.1, 19.0, 18.4, 14.0, 29.5] |
| 28 | 0-8010 | 4023 | rally | serve_start | `(1210,923)` |  | low=(1210, 1014) apex=(1210, 923) rise=91px up_steps=3 |
| 29 | 0-8010 | 4196 | rally | point_end | `(2302,463)` | 1.0 | reason=Ball hit the net duration=173f vel_hist=[9.1, 4.1, 5.4, 5.0, 1.0] |
| 30 | 0-8010 | 4784 | rally | serve_start | `(2582,890)` |  | low=(2557, 1020) apex=(2582, 890) rise=130px up_steps=3 |
| 31 | 0-8010 | 4888 | rally | point_end | `(3830,2146)` | 59.5 | reason=Ball bounce outside singles court (right sideline) duration=104f vel_hist=[68.4, 65.7, 62.6, 61.3, 59.5] |
| 32 | 0-8010 | 5224 | rally | serve_start | `(1266,868)` |  | low=(1274, 993) apex=(1266, 868) rise=125px up_steps=3 |
| 33 | 0-8010 | 5439 | offscreen | upper_side_exit | `(1678,39)` |  | DEBUG: Ball likely exited through upper side from (1678,39) |
| 34 | 0-8010 | 5439 | offscreen | top_return_wait_activated | `(1678,39)` |  | DEBUG: [TOP-RETURN WAIT] activated for delayed upper-side re-entry search |
| 35 | 0-8010 | 5479 | onscreen | top_return_reentry | `(1926,112)` |  | DEBUG: [TOP-RETURN WAIT] prioritizing re-entry candidate at (1926, 112) area=162.5px score=1136.8 motion=78.8/240.0 source=primary |
| 36 | 0-8010 | 5481 | onscreen | top_return_continuation | `(1952,280)` |  | DEBUG: [TOP-RETURN GRACE] prioritizing continuation candidate at (1952, 280) area=204.5px score=71.6 motion=53.7/102.0 source=primary |
| 37 | 0-8010 | 5482 | onscreen | top_return_continuation | `(1966,381)` |  | DEBUG: [TOP-RETURN GRACE] prioritizing continuation candidate at (1966, 381) area=209.5px score=86.5 motion=45.1/96.0 source=alt |
| 38 | 0-8010 | 5483 | onscreen | top_return_continuation | `(1980,490)` |  | DEBUG: [TOP-RETURN GRACE] prioritizing continuation candidate at (1980, 490) area=219.5px score=87.2 motion=60.7/195.0 source=primary |
| 39 | 0-8010 | 5519 | offscreen | back_bottom_exit | `(2924,2112)` |  | DEBUG: Ball near back/bottom exit (2924,2112), may have gone off-screen |
| 40 | 0-8010 | 5519 | offscreen | back_return_wait_activated | `(2924,2112)` |  | DEBUG: [BACK-RETURN WAIT] activated for delayed back-screen re-entry search |
| 41 | 0-8010 | 5524 | onscreen | back_return_reentry | `(2900,1982)` |  | DEBUG: [BACK-RETURN WAIT] prioritizing re-entry candidate at (2900, 1982) area=952.5px score=1048.8 motion=67.8/149.0 source=primary |
| 42 | 0-8010 | 5525 | onscreen | back_return_reentry | `(2864,1638)` |  | DEBUG: [BACK-RETURN WAIT] prioritizing re-entry candidate at (2864, 1638) area=1054.0px score=736.9 motion=51.7/108.0 source=primary |
| 43 | 0-8010 | 5526 | onscreen | back_return_reentry | `(2813,1319)` |  | DEBUG: [BACK-RETURN WAIT] prioritizing re-entry candidate at (2813, 1319) area=1019.5px score=332.9 motion=45.9/85.0 source=primary |
| 44 | 0-8010 | 5527 | onscreen | back_return_reentry | `(2753,1039)` |  | DEBUG: [BACK-RETURN WAIT] prioritizing re-entry candidate at (2753, 1039) area=763.5px score=329.0 motion=46.6/100.0 source=primary |
| 45 | 0-8010 | 5536 | offscreen | upper_side_exit | `(2393,33)` |  | DEBUG: Ball likely exited through upper side from (2393,33) |
| 46 | 0-8010 | 5536 | offscreen | top_return_wait_activated | `(2393,33)` |  | DEBUG: [TOP-RETURN WAIT] activated for delayed upper-side re-entry search |
| 47 | 0-8010 | 5557 | onscreen | top_return_reentry | `(2146,67)` |  | DEBUG: [TOP-RETURN WAIT] prioritizing re-entry candidate at (2146, 67) area=22.5px score=728.3 motion=28.9/160.0 source=primary |
| 48 | 0-8010 | 5558 | onscreen | top_return_continuation | `(2141,86)` |  | DEBUG: [TOP-RETURN DOWNWARD] prioritizing near vertical return at (2141, 86) area=38.0px score=490.7 motion=30.7/145.0 source=primary |
| 49 | 0-8010 | 5559 | onscreen | top_return_continuation | `(2138,109)` |  | DEBUG: [TOP-RETURN GRACE] prioritizing continuation candidate at (2138, 109) area=43.5px score=154.5 motion=27.3/145.0 source=primary |
| 50 | 0-8010 | 5560 | onscreen | top_return_continuation | `(2134,131)` |  | DEBUG: [TOP-RETURN GRACE] prioritizing continuation candidate at (2134, 131) area=18.0px score=175.4 motion=25.2/145.0 source=primary |
| 51 | 0-8010 | 5561 | onscreen | top_return_continuation | `(2130,152)` |  | DEBUG: [TOP-RETURN GRACE] prioritizing continuation candidate at (2130, 152) area=21.0px score=157.1 motion=21.2/185.0 source=alt |
| 52 | 0-8010 | 5765 | rally | point_end | `` |  | reason=POINT_TIMEOUT duration=541f |
| 53 | 0-8010 | 5913 | rally | serve_start | `(2625,887)` |  | low=(2588, 1024) apex=(2625, 887) rise=137px up_steps=3 |
| 54 | 0-8010 | 5945 | rally | point_end | `(1797,442)` | 9.1 | reason=Ball hit the net duration=32f vel_hist=[36.9, 34.4, 28.2, 10.3, 9.1] |
| 55 | 0-8010 | 6023 | rally | serve_start | `(2432,915)` |  | low=(2408, 1003) apex=(2432, 915) rise=88px up_steps=3 |
| 56 | 0-8010 | 6127 | offscreen | top_exit_projected | `(1313,55)` |  | DEBUG: Ball projected off TOP edge from (1313,55) |
| 57 | 0-8010 | 6127 | offscreen | top_return_wait_activated | `(1313,55)` |  | DEBUG: [TOP-RETURN WAIT] activated for projected top-edge exit |
| 58 | 0-8010 | 6159 | onscreen | top_return_reentry | `(2039,35)` |  | DEBUG: [TOP-RETURN WAIT] prioritizing re-entry candidate at (2039, 35) area=10.5px score=1762.5 motion=31.5/204.0 source=primary |
| 59 | 0-8010 | 6163 | onscreen | top_return_continuation | `(2067,127)` |  | DEBUG: [TOP-RETURN DOWNWARD] prioritizing near vertical return at (2067, 127) area=5.5px score=350.3 motion=16.0/139.0 source=primary |
| 60 | 0-8010 | 6166 | rally | point_end | `(2079,176)` | 25.7 | reason=Ball bounced out of court (far baseline) duration=143f vel_hist=[0.0, 184.1, 25.0, 24.7, 25.7] |
| 61 | 0-8010 | 6668 | rally | serve_start | `(1194,868)` |  | low=(1200, 1012) apex=(1194, 868) rise=144px up_steps=3 |
| 62 | 0-8010 | 6870 | offscreen | top_exit_clipped | `(1853,4)` |  | DEBUG: Ball clipped/off TOP edge from (1853,4) |
| 63 | 0-8010 | 6870 | offscreen | top_return_wait_activated | `(1853,4)` |  | DEBUG: [TOP-RETURN WAIT] activated for shared top-edge re-entry search |
| 64 | 0-8010 | 6876 | onscreen | top_return_reentry | `(1938,2)` |  | DEBUG: [TOP-RETURN WAIT] prioritizing re-entry candidate at (1938, 2) area=33.5px score=210.0 motion=40.9/242.0 source=primary |
| 65 | 0-8010 | 6877 | onscreen | top_return_continuation | `(1947,9)` |  | DEBUG: [TOP-RETURN GRACE] prioritizing continuation candidate at (1947, 9) area=28.0px score=134.0 motion=27.2/206.0 source=alt |
| 66 | 0-8010 | 6878 | onscreen | top_return_continuation | `(1956,17)` |  | DEBUG: [TOP-RETURN GRACE] prioritizing continuation candidate at (1956, 17) area=41.0px score=-12.5 motion=32.1/240.0 source=primary |
| 67 | 0-8010 | 6879 | onscreen | top_return_continuation | `(1964,26)` |  | DEBUG: [TOP-RETURN GRACE] prioritizing continuation candidate at (1964, 26) area=41.5px score=-26.6 motion=37.5/237.0 source=primary |
| 68 | 0-8010 | 6880 | onscreen | top_return_continuation | `(1971,37)` |  | DEBUG: [TOP-RETURN GRACE] prioritizing continuation candidate at (1971, 37) area=37.5px score=-20.9 motion=33.9/241.0 source=primary |
| 69 | 0-8010 | 7052 | rally | point_end | `` |  | reason=POINT_TIMEOUT duration=384f |
| 70 | 0-8010 | 7282 | rally | serve_start | `(2631,885)` |  | low=(2595, 1012) apex=(2631, 885) rise=127px up_steps=3 |
| 71 | 0-8010 | 7383 | offscreen | top_exit_projected | `(1397,52)` |  | DEBUG: Ball projected off TOP edge from (1397,52) |
| 72 | 0-8010 | 7383 | offscreen | top_return_wait_activated | `(1397,52)` |  | DEBUG: [TOP-RETURN WAIT] activated for projected top-edge exit |
| 73 | 0-8010 | 7415 | onscreen | top_return_reentry | `(2036,29)` |  | DEBUG: [TOP-RETURN WAIT] prioritizing re-entry candidate at (2036, 29) area=25.5px score=1639.1 motion=33.3/227.0 source=primary |
| 74 | 0-8010 | 7417 | onscreen | top_return_continuation | `(2049,73)` |  | DEBUG: [TOP-RETURN GRACE] prioritizing continuation candidate at (2049, 73) area=10.5px score=4.7 motion=22.9/142.0 source=alt |
| 75 | 0-8010 | 7418 | onscreen | top_return_continuation | `(2056,97)` |  | DEBUG: [TOP-RETURN GRACE] prioritizing continuation candidate at (2056, 97) area=16.5px score=164.9 motion=21.3/149.0 source=primary |
| 76 | 0-8010 | 7419 | onscreen | top_return_continuation | `(2062,121)` |  | DEBUG: [TOP-RETURN GRACE] prioritizing continuation candidate at (2062, 121) area=19.5px score=175.7 motion=45.8/146.0 source=alt |
| 77 | 0-8010 | 7456 | rally | point_end | `(3209,1177)` | 151.3 | reason=Ball bounced out of court (right sideline) duration=174f vel_hist=[57.0, 63.0, 67.2, 73.1, 151.3] |
| 78 | 0-8010 | 7769 | rally | serve_start | `(1223,880)` |  | low=(1234, 1022) apex=(1223, 880) rise=142px up_steps=3 |
| 79 | 0-8010 | 7885 | rally | point_end | `(1612,441)` | 25.0 | reason=Ball hit the net duration=116f vel_hist=[39.8, 33.4, 27.8, 24.0, 25.0] |
| 80 | 0-8010 | 7986 | rally | serve_start | `(1239,715)` |  | low=(1375, 854) apex=(1239, 715) rise=139px up_steps=2 |
| 81 | 0-8010 | 8001 | rally | point_end | `(1213,530)` | 14.9 | reason=Ball bounced out of court (left sideline) duration=15f vel_hist=[65.5, 57.4, 10.3, 39.6, 14.9] |
| 82 | 9600-10100 | 9639 | rally | serve_start | `(2234,833)` |  | low=(2218, 991) apex=(2234, 833) rise=158px up_steps=3 |
| 83 | 9600-10100 | 9750 | offscreen | upper_side_exit | `(1622,40)` |  | DEBUG: Ball likely exited through upper side from (1622,40) |
| 84 | 9600-10100 | 9750 | offscreen | top_return_wait_activated | `(1622,40)` |  | DEBUG: [TOP-RETURN WAIT] activated for delayed upper-side re-entry search |
| 85 | 9600-10100 | 9781 | onscreen | top_return_reentry | `(2206,61)` |  | DEBUG: [TOP-RETURN WAIT] prioritizing re-entry candidate at (2206, 61) area=23.0px score=1807.9 motion=30.6/234.0 source=primary |
| 86 | 9600-10100 | 9783 | onscreen | top_return_continuation | `(2218,107)` |  | DEBUG: [TOP-RETURN GRACE] prioritizing continuation candidate at (2218, 107) area=14.0px score=175.2 motion=27.7/162.0 source=alt |
| 87 | 9600-10100 | 9784 | onscreen | top_return_continuation | `(2224,131)` |  | DEBUG: [TOP-RETURN GRACE] prioritizing continuation candidate at (2224, 131) area=10.0px score=183.8 motion=22.4/147.0 source=alt |
| 88 | 9600-10100 | 9785 | onscreen | top_return_continuation | `(2229,154)` |  | DEBUG: [TOP-RETURN GRACE] prioritizing continuation candidate at (2229, 154) area=16.5px score=169.1 motion=23.6/136.0 source=alt |
| 89 | 9600-10100 | 9970 | rally | point_end | `` |  | reason=POINT_TIMEOUT duration=331f |
