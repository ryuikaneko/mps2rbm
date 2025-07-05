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

    ranks = np.arange(8,65,8)
    dat1 = []
    for i,rank in enumerate(ranks):
        dir = "../L16g1.0_D8_rank"+"{}".format(rank)+"/input/"
        file = "dat_cp_enes_L16_g1.0000000000_D8_rank"+"{}".format(rank)+"_seed12345"
        dat1.append(np.loadtxt(dir+file,dtype=np.complex128))
    dat1 = np.array(dat1)
#    print(*(d.shape for d in dat1))
#    print(dat1[:,3])
#    print(dat1[:,5])
#    print(dat1[:,7])
#    print(1-np.abs(dat1[:,7]/dat1[:,5]))
#    print(dat1[:,9])
    dat2 = []
    for i,rank in enumerate(ranks):
        dir = "../L16g1.0_D8_rank"+"{}".format(rank)+"/"
        file = "dat_vmc_enes_ave_err_L16_g1.0000000000_D8_rank"+"{}".format(rank)+"_seed12345"
        dat2.append(np.loadtxt(dir+file,dtype=np.complex128))
    dat2 = np.array(dat2)
#    print(*(d.shape for d in dat2))
#    print(dat2[:,3])
#    print(dat2[:,8])
#    print(dat2[:,9])
#    print(1-np.abs(dat2[:,8]/dat2[:,5]))
#    print(dat2[:,10])
#    print(np.abs(dat2[:,9]/dat2[:,5]))
#    print(dat2[:,11])
    dat3 = []
    for i,rank in enumerate(ranks):
        dir = "../L16g1.0_D8_rank"+"{}".format(rank)+"_periodic/"
        file = "dat_vmc_enes_ave_err_L16_g1.0000000000_D8_rank"+"{}".format(rank)+"_seed12345"
        dat3.append(np.loadtxt(dir+file,dtype=np.complex128))
    dat3 = np.array(dat3)
#    print(*(d.shape for d in dat3))
#    print(dat3[:,3])
#    print(dat3[:,8])
#    print(dat3[:,9])
#    print(1-np.abs(dat3[:,8]/dat3[:,5]))
#    print(dat3[:,10])
#    print(np.abs(dat3[:,9]/dat3[:,5]))
#    print(dat3[:,11])
    dat = np.array([
        dat1[:,3].real,
        1-np.abs(dat1[:,7]/dat1[:,5]),
        1-np.abs(dat2[:,8]/dat2[:,5]),
        np.abs(dat2[:,9]/dat2[:,5]),
        1-np.abs(dat3[:,8]/dat3[:,5]),
        np.abs(dat3[:,9]/dat3[:,5]),
        ]).T
    print(dat)
    print(dat[:,0])

    plt.figure(figsize=(10,5))
    plt.xlabel(r"$1/R$")
    plt.ylabel(r"error")
#    plt.xlim(-0.005,0.130)
#    plt.ylim(-0.005,0.06)
    plt.xlim(1e-2,0.2)
    plt.ylim(1e-6,1)
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(which="both")
    xx = np.linspace(1e-2,1,129)
#    yy = xx**2
    yy = 100*xx**2
    plt.plot(xx,yy,ls="--",marker="none",lw=3,color="black",label=r"$\propto R^{-2}$")
    plt.plot(1.0/dat[:,0],dat[:,1],
        ls="none",
        marker="o",
        color="darkred",
        mfc="darkorange",
        ms=14,
        mew=3,
        alpha=0.75,
        label=r"before VMC (open)",
        )
    plt.errorbar(1.0/dat[:,0],dat[:,2],yerr=dat[:,3],
        ls="none",
        elinewidth=3,
        capsize=4,
        fmt="v",
        color="darkgreen",
        mfc="white",
        ms=14,
        mew=3,
        alpha=0.75,
        label=r"after VMC (open)",
        )
    plt.errorbar(1.0/dat[:,0],dat[:,4],yerr=dat[:,5],
        ls="none",
        elinewidth=3,
        capsize=4,
        fmt="^",
        color="darkviolet",
        mfc="white",
        ms=14,
        mew=3,
        alpha=0.75,
        label=r"after VMC (periodic)",
        )
    plt.text(-0.125,0.85,r"(b)",fontsize=36,ha="center",va="center",transform=plt.gca().transAxes)
    plt.legend(ncols=1,columnspacing=1,handletextpad=0,fontsize=24,loc="lower right")
    plt.savefig("fig_ene_errors_vs_rank_log.pdf",bbox_inches="tight")
    plt.close()

if __name__ == '__main__':
    main()

