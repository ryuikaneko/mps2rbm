import numpy as np
import matplotlib.pyplot as plt

def main():
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.major.pad"] = 8
    plt.rcParams["ytick.major.pad"] = 8
    plt.rcParams["xtick.major.width"] = 1.5
    plt.rcParams["ytick.major.width"] = 1.5
    plt.rcParams["xtick.major.size"] = 8
    plt.rcParams["ytick.major.size"] = 8
    plt.rcParams["font.size"] = 30
    plt.rcParams["axes.linewidth"] = 2

    dir = "../L16g0.5_D8/"
#    ranks = np.arange(8,65,8)
#    ranks = np.array([8,16,32,48,64])
    ranks = np.array([8,16,32,64])
    seeds = np.arange(12340,12350)
    dat = []
    for rank in ranks:
        for seed in seeds:
            file = dir + "dat_cp_errors_L16_g0.5000000000_D8_rank{}_seed{}".format(rank, seed)
            dat.append(np.loadtxt(file))
    dat = np.array(dat)
    print(dat.shape)

#    plt.figure(figsize=(10,5))
    plt.figure(figsize=(10,8))
    plt.xlabel(r"iteration")
    plt.ylabel(r"infidelity")
#    plt.xscale("log")
    plt.yscale("log")
#    plt.xlim(10,1e3)
    plt.xlim(0,1e3)
#    plt.ylim(3e-3,2e-1)
    plt.ylim(1.5e-5,2e-1)
    plt.grid(which="both")
    markers = ["o","v","^","<",">","s","p","*","x","D"][::-1]
    cmap = plt.get_cmap("gist_earth")
    skip = 50
    cnt = 0
    for irank,rank in enumerate(ranks):
        for iseed,seed in enumerate(seeds):
            if iseed==len(seeds)-1:
#            if iseed==0:
                label1 = r"$R={}$".format(rank)
            else:
                label1 = ""
#            if irank==len(ranks)-1:
#                dashes1 = [1,0]
#            else:
#                dashes1 = [irank+1,irank+1]
            dashes1 = [1,0]
            alpha1 = 1.0-0.6*(len(ranks)-(irank+1.0))/len(ranks)
            plt.plot(np.arange(len(dat[cnt])),dat[cnt],
                ls="-",
                dashes=dashes1,
                marker="none",
                color=cmap(0.6-0.6*(irank+1.0)/len(ranks)),
                alpha=alpha1,
                label=""
                )
            cnt += 1
    cnt = 0
    for irank,rank in enumerate(ranks):
        for iseed,seed in enumerate(seeds):
            if iseed==len(seeds)-1:
#            if iseed==0:
                label1 = r"$R={}$".format(rank)
            else:
                label1 = ""
            plt.plot(skip*np.arange(len(dat[cnt,5::skip])),dat[cnt,5::skip],
                ls="none",
                marker=markers[iseed],
                color=cmap(0.6-0.6*(irank+1.0)/len(ranks)),
                mfc="white",
                ms=8,
                mew=2,
                label=label1
                )
            cnt += 1
#    plt.legend(ncols=1,columnspacing=1,handletextpad=0,fontsize=24,bbox_to_anchor=(1,1.03))
    plt.legend(ncols=4,columnspacing=1,handletextpad=0,fontsize=22,loc="upper center")
#    plt.legend(ncols=3,columnspacing=1,handletextpad=0,fontsize=22,loc="upper center")
    plt.savefig("fig_fidelity_errors_vs_steps.pdf",bbox_inches="tight")
    plt.close()

if __name__ == '__main__':
    main()

