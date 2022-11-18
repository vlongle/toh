import matplotlib.pyplot as plt
import seaborn as sns

N = [1, 5, 10, 20, 50, 100]
zeroshot_ratio = [1., 1.05319595, 1.64417358,
                  2.05844852, 2.41852224, 2.37134959]
final_perf_ratio = [1, 1.40417475, 1.4797179, 1.54266, 1.47910783, 1.56442257]

num_steps_ratio = [1., 1.28712771, 1.46067302,
                   1.28712771, 1.3131303, 1.91176321]

sns.set_style("whitegrid")

#sns.lineplot(x=N, y=zeroshot_ratio)
#sns.lineplot(x=N, y=final_perf_ratio)
sns.lineplot(x=N, y=num_steps_ratio)
# x-label: No. agents, y-label: Zero-shot ratio
plt.xlabel("No. agents")
#plt.ylabel("Zero-shot ratio")
#plt.ylabel("Final performance ratio")
plt.ylabel("Reversed No. steps to threshold ratio")
plt.xlim(0, 100)
# plot a red horizontal line at 1.3
plt.axhline(y=2.0, color="r", linestyle="--")
# plt.savefig("zeroshot_ratio.png")
# plt.savefig("final_perf_ratio.png")
plt.savefig("num_steps_ratio.png")
