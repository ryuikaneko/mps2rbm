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

    dir = "../L16g1.0_D8/"
    ranks = np.arange(8,65,8)
    seeds = np.arange(12340,12350)
    dat = []
    for rank in ranks:
        for seed in seeds:
            file = dir + "dat_cp_enes_L16_g1.0000000000_D8_rank{}_seed{}".format(rank, seed)
            dat.append(np.loadtxt(file))
    dat = np.array(dat)
    print(dat.shape)
    print(dat[:,3])
    print(dat[:,9])
    print(dat[:,10])

    plt.figure(figsize=(10,5))
    plt.xlabel(r"$1/R$")
    plt.ylabel(r"error")
#    plt.xlim(-0.005,0.130)
#    plt.ylim(-0.01,0.15)
    plt.xlim(1e-2,0.2)
#    plt.ylim(1e-3,1)
    plt.ylim(1e-6,1)
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(which="both")
    markers = ["o","v","^","<",">","s","p","*","x","D"][::-1]
    xx = np.linspace(1e-2,1,129)
#    yy = xx**2
    yy = 100*xx**2
    plt.plot(xx,yy,ls="--",marker="none",lw=3,color="black",label=r"$\propto R^{-2}$")
    cnt = 0
    for irank,rank in enumerate(ranks):
        for iseed,seed in enumerate(seeds):
            if irank==len(ranks)-1 and iseed==len(seeds)-1:
                label1 = r"infidelity"
            else:
                label1 = "" 
            plt.plot(1.0/dat[cnt,3],dat[cnt,10],
                ls="none",
                marker=markers[iseed],
                color="darkblue",
                mfc="white",
                ms=18,
                mew=3,
                alpha=0.75,
                label=label1
                )
            cnt += 1
    cnt = 0
    for irank,rank in enumerate(ranks):
        for iseed,seed in enumerate(seeds):
            if irank==len(ranks)-1 and iseed==len(seeds)-1:
                label1 = r"energy error"
            else:
                label1 = "" 
            plt.plot(1.0/dat[cnt,3],dat[cnt,9],
                ls="none",
                marker=markers[iseed],
                color="darkred",
                mfc="darkorange",
                ms=14,
                mew=3,
                alpha=0.75,
                label=label1
                )
            cnt += 1
    plt.text(-0.125,0.85,r"(b)",fontsize=36,ha="center",va="center",transform=plt.gca().transAxes)
    plt.legend(ncols=1,columnspacing=1,handletextpad=0,fontsize=24,loc="lower right")
    plt.savefig("fig_ene_errors_vs_rank_log.pdf",bbox_inches="tight")
    plt.close()

if __name__ == '__main__':
    main()

