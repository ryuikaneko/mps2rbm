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
    dat = []
    for i,rank in enumerate(ranks):
        dir = "../L16g1.0_D8_rank"+"{}".format(rank)+"_periodic/"
        file = "dat_vmc_enes_opt_L16_g1.0000000000_D8_rank"+"{}".format(rank)+"_seed12345"
        dat.append(np.loadtxt(dir+file,dtype=np.complex128))
    print(*(d.shape for d in dat))

#    plt.figure(figsize=(10,5))
    plt.figure(figsize=(10,8))
    plt.xlabel(r"iteration")
    plt.ylabel(r"energy")
    plt.xlim(0,2000)
    plt.ylim(-20.5,-19.3)
    plt.grid()
    cmap = plt.get_cmap("gist_earth")
    for i,d in enumerate(dat):
        alpha1 = 1.0-0.1*(len(dat)-(i+1.0))/len(dat)
        plt.plot(np.arange(d.shape[0]),np.real(d),
            label=r"$R="+str(ranks[i])+"$",
            color=cmap(0.6-0.6*(i+1.0)/len(dat)),
            alpha=alpha1,
        )
#    e_exact=-2.001638790048517436e+01
    e_exact=-2.040459447475665655e+01
    plt.plot([0,2000],[e_exact,e_exact],ls="--",lw=3,color="darkred",label=r"exact")
    plt.legend(ncols=3,columnspacing=1,handletextpad=0.2,fontsize=22,loc="upper right")
    plt.savefig("fig_optimization_periodic.pdf",bbox_inches="tight")
    plt.close()

if __name__ == '__main__':
    main()

