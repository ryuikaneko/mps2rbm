import numpy as np
import matplotlib.pyplot as plt
import glob

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

    Ls = np.array([16,18,20,22,24,28,32,40,48])
#    Ls = np.array([16,18,20,22,24,28,32,48])
#    Ls = np.array([16,18,20,22,24,28,32,40,48,64,80])
    Ds = Ls//2
    ranks = Ls//2
    dat1 = []
    for i,L in enumerate(Ls):
        dir = "../mps_rbm/L"+"{}".format(L)+"g2.0_D"+"{}".format(Ds[i])+"_rank"+"{}".format(ranks[i])+"_periodic_1e4plus1e5samples/"
        file = "dat_vmc_enes_ave_err_L"+"{}".format(L)+"_g2.0000000000_D"+"{}".format(Ds[i])+"_rank"+"{}".format(ranks[i])+"_seed*"
        filename = glob.glob(dir+file)[0]
        print(filename)
        dat1.append(np.loadtxt(filename,dtype=np.complex128))
#        dir = "../mps_rbm/L"+"{}".format(L)+"g2.0_D"+"{}".format(Ds[i])+"_rank"+"{}".format(ranks[i])+"_periodic_1e4plus1e5samples/"
#        file = "dat_vmc_enes_ave_err_L"+"{}".format(L)+"_g2.0000000000_D"+"{}".format(Ds[i])+"_rank"+"{}".format(ranks[i])+"_seed12345"
#        dat1.append(np.loadtxt(dir+file,dtype=np.complex128))
    dat1 = np.array(dat1).real
    print(dat1)
    print(dat1.shape)

    plt.figure(figsize=(10,5))
    plt.xlabel(r"$1/n^2$")
    plt.ylabel(r"energy per site")
    plt.xlim(-0.0002,0.0042)
    plt.ylim(-2.2,-1.9)
#    plt.ylim(-2.2,-1.95)
#    plt.ylim(-1.28,-1.22)
    plt.xticks([0,0.001,0.002,0.003,0.004])
#    plt.yticks([-1.28,-1.26,-1.24,-1.22])
    plt.grid()
#
    p = np.poly1d(np.polyfit( 1.0/dat1[:,0]**2, dat1[:,5]/dat1[:,0], 1 ))
    xx = np.linspace(-0.0002,0.0042,33)
#    yy = -4.0/np.pi-np.pi/(6.0)*xx
    yy = p(xx)
#
    plt.plot(xx,yy,
        ls="--",
        lw=3,
        marker="none",
        color="darkred",
        label=r"fit",
#        label=r"asymptote",
        )
    plt.plot(xx,0.96*yy,
#    plt.plot(xx,0.97*yy,
        ls=":",
        lw=3,
        marker="none",
        color="darkred",
        label=r"$4\%$ error",
#        label=r"$3\%$ error",
        )
    plt.plot(1.0/dat1[:,0]**2,dat1[:,5]/dat1[:,0],
        ls="none",
        marker="o",
        color="darkred",
        mfc="white",
        ms=14,
        mew=3,
        label=r"exact (periodic)",
        )
    plt.errorbar(1.0/dat1[:,0]**2,dat1[:,7]/dat1[:,0],yerr=dat1[:,8]/dat1[:,0],
        ls="none",
        elinewidth=3,
        capsize=4,
        fmt="^",
        color="darkviolet",
        mfc="white",
        ms=14,
        mew=3,
        alpha=0.75,
        label=r"VMC (periodic)",
        )
    ## https://www.geeksforgeeks.org/python/how-to-change-order-of-items-in-matplotlib-legend/
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [3, 2, 1, 0]
    plt.legend(
        [handles[i] for i in order], [labels[i] for i in order],
        ncols=2,columnspacing=2,handletextpad=.75,fontsize=24,loc="upper right",
        )
    plt.savefig("fig.pdf",bbox_inches="tight")
    plt.close()

if __name__ == '__main__':
    main()

