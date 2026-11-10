# Measured head regions with person fallback

Registered before this comparison. Direct person-box upper views fail on an
upside-down gymnast, while existing local face models locate the face. Test
measured head locations instead of assuming every detected body is upright.
This is a mechanism diagnostic informed by reused DEV pixels, not independent
confirmation or a new artistic label set.

Reuse all72 source identities, original seven face models, pose-derived head
estimates and Lite2 person boxes. Source-only proposals never use nominated
heads, selected-subject IDs, crop windows or judgments. Two fixed policies:
faces with body fallback; faces+pose with body fallback (primary). Pad measured
head boxes25 percent by the existing helper. Match a measured unpadded head
to a padded full-person box when its center lies inside that body box. For
bodies with any matched measured head, do not add an upright upper-box guess.
For unmatched bodies add the existing padded upper45-percent view. Keep all
measured heads, including those not assigned to a detected body; deduplicate
only exact boxes. Missing detections do not certify absence.

Reuse the exact five frozen parents and interval solver from the previous
study. All heads/fallback regions in the outer must remain in strict inner;
other regions can be completely excluded. Minimal retains maximum extent.
No safe outer means no pair; no strict inner means Minimal alone. No supplied
regions retains the unchecked parent explicitly. Ten controls, no winner
chosen on DEV. Report all139 accepted essential DEV endpoints alongside the
full396 head-known cohort, total offer coverage, contain emissions, old-match
loss, common-offer repairs/new cuts, FIT and secondary results. Reproduce all
parent diagnostic rows before applying constraints.

No new training, inference, paid API calls, TEST, label admission, export or
promotion. Compare the known gymnast mechanism and a remaining failure with
actual owned pixels, with full-source context and independently decoded PNGs.
Original frozen-bank quality has not been qualified by this geometry screen.
Run-heavy4GiB/two threads;16MiB packet, including bounded witnesses.
