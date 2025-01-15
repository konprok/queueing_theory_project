import numpy as np
import matplotlib.pyplot as plt
from math import factorial, exp
from config import CONFIG

def build_routing_matrix_4d(cfg):
    """
    Buduje 4D p_{(i,r)->(j,s)} dla N=7 węzłów:
      0=registration,
      1=admissions,
      2=gynecology,
      3=predelivery,
      4=delivery,
      5=icu (OIOM),
      6=postpartum.

    r=0 (kl1=bez komplik.), r=1 (kl2=z komplik.), r=2 (kl3=krytyczne).
    """

    N=7
    R=3
    P = np.zeros((N,R,N,R))

    for r_ in range(R):
        P[0, r_, 1, r_] = 1.0

    p_a10_pre = cfg['admissions_class1_predelivery']
    p_a10_del = cfg['admissions_class1_delivery']
    P[1,0,3,0] = p_a10_pre
    P[1,0,4,0] = p_a10_del

    p_a12_pre = cfg['admissions_class2_predelivery']
    p_a12_del = cfg['admissions_class2_delivery']
    p_a12_gyn = cfg['admissions_class2_gyn']
    P[1,1,3,1] = p_a12_pre
    P[1,1,4,1] = p_a12_del
    P[1,1,2,1] = p_a12_gyn

    P[1,2,4,2] = 1.0

    P[3,0,4,0] = 1.0

    P[3,1,4,1] = 1.0

    P[3,2,4,0] = 0.5
    P[3,2,4,1] = 0.5

    p_3to2 = cfg['delivery_class3_to2']
    p_3to1 = cfg['delivery_class3_to1']

    P[4,2,5,2] = 0.3
    P[4,2,6,1] = p_3to2
    P[4,2,6,0] = p_3to1

    p_2to1 = cfg['delivery_class2_to1']
    p_2to3 = cfg['delivery_class2_to3']
    P[4,1,5,2] = p_2to3
    P[4,1,6,0] = p_2to1
    P[4,1,6,1] = 1.0 - (p_2to1 + p_2to3)

    p_1to3= cfg['delivery_class1_to3']
    p_1to2= cfg['delivery_class1_to2']
    P[4,0,5,2] = p_1to3
    P[4,0,6,1] = p_1to2
    P[4,0,6,0] = 1.0 - (p_1to3 + p_1to2)

    P[5,2,6,0] = 0.5

    return P


def _mmc_stationary_distribution(c, rho, max_k=20):
    if rho>=c:
        return None
    Z=0.0
    from math import factorial
    for k in range(c):
        Z += (rho**k)/factorial(k)
    Z += (rho**c/factorial(c))*( c/(c-rho) )
    p0=1.0/Z
    dist=[]
    for k_ in range(max_k+1):
        if k_<c:
            val = p0*(rho**k_)/factorial(k_)
        else:
            val = p0*(rho**c/factorial(c))*((1.0/c)**(k_-c))
        dist.append(val)
    return dist

def compute_bcmp_open_network_4d(
    N, R,
    arrival_rates,
    service_rates,
    node_types,
    servers_per_node,
    routing_matrix_4d,
    max_k_for_print=12
):
    """
    BCMP z class switching (p_{(i,r)->(j,s)}).
    Zwraca { 'e_i_r','rho_i_r','rho_i','K_i_r','K_i','pi_i_k' }
    """
    # p0[i,r]
    p0 = np.zeros((N,R))
    # kl0, kl1 => node0, kl2 => node1
    p0[0,0] = 1.0
    p0[0,1] = 1.0
    p0[1,2] = 1.0

    # 1) e_{i,r}
    e_i_r = np.copy(p0)
    for _ in range(1000):
        new_e = e_i_r.copy()
        for i in range(N):
            for r_ in range(R):
                external_inflow = arrival_rates[r_]*p0[i,r_]
                internal_inflow = 0.0
                for j in range(N):
                    for s in range(R):
                        internal_inflow += e_i_r[j,s]*routing_matrix_4d[j,s,i,r_]
                new_e[i,r_] = external_inflow + internal_inflow
        if np.allclose(new_e, e_i_r, atol=1e-12):
            break
        e_i_r = new_e

    # 2) rho_{i,r} = (λ_r * e_{i,r}) / μ_{i,r}
    rho_i_r = np.zeros((N,R))
    for i in range(N):
        for r_ in range(R):
            mu_ = service_rates[i,r_]
            if mu_>0:
                rho_i_r[i,r_] = (arrival_rates[r_]* e_i_r[i,r_]) / mu_
    rho_i = np.sum(rho_i_r, axis=1)

    # 3) K_i_r (średnie liczby w węzłach)
    K_i_r = np.zeros((N,R))
    for i in range(N):
        typ = node_types[i]
        if typ in [1,4]:  # M/M/1 lub PS
            if rho_i[i]<1.0:
                # Rozkład proporcjonalny do rho_{i,r}
                denom= rho_i[i]
                for r_ in range(R):
                    if denom>0:
                        alpha= rho_i_r[i,r_]/ denom
                        K_i_r[i,r_]= alpha*(denom/(1-denom))
        elif typ==2:  # M/M/c
            c_ = servers_per_node[i]
            lam_i = sum(arrival_rates[r_]* e_i_r[i,r_] for r_ in range(R))
            mus = [service_rates[i,xx] for xx in range(R) if service_rates[i,xx]>0]
            if len(mus)==0:
                continue
            mu_avg= np.mean(mus)
            rho_prime= lam_i/(c_*mu_avg)
            if rho_prime<1.0:
                # ErlangC
                def erlang_c(c,a):
                    from math import factorial
                    num= (a**c)/factorial(c)*( c/(c-a) )
                    ss=0.0
                    for k_ in range(c):
                        ss+=(a**k_)/factorial(k_)
                    ss+= num
                    return num*(1.0/ss)
                a= lam_i/mu_avg
                C= erlang_c(c_,a)
                Lq= C*(rho_prime/(1-rho_prime))
                Ls= c_*rho_prime
                L= Lq+Ls
                sum_rhoi= sum(rho_i_r[i,:])
                for r_ in range(R):
                    if sum_rhoi>0:
                        alpha= rho_i_r[i,r_]/ sum_rhoi
                        K_i_r[i,r_]= alpha*L
        elif typ==3:  # M/M/∞
            for r_ in range(R):
                K_i_r[i,r_]= rho_i_r[i,r_]

    K_i= np.sum(K_i_r,axis=1)

    # 4) pi_i_k (rozkłady stacjonarne)
    pi_i_k= {}
    maxk= max_k_for_print
    for i in range(N):
        typ= node_types[i]
        if typ in [1,4]:  # M/M/1 lub PS
            if rho_i[i]<1.0:
                r_= rho_i[i]
                dist=[]
                for k_ in range(maxk+1):
                    dist.append( (1-r_)* (r_**k_) )
                pi_i_k[i]= np.array(dist)
            else:
                pi_i_k[i]= None
        elif typ==2: # M/M/c
            c_= servers_per_node[i]
            lam_i= sum(arrival_rates[r_]* e_i_r[i,r_] for r_ in range(R))
            mus = [service_rates[i,xx] for xx in range(R) if service_rates[i,xx]>0]
            if len(mus)==0:
                pi_i_k[i]= None
                continue
            mu_avg= np.mean(mus)
            bigrho= lam_i/(mu_avg)
            if bigrho>= c_:
                pi_i_k[i]= None
            else:
                dist= _mmc_stationary_distribution(c_, bigrho, maxk)
                pi_i_k[i]= np.array(dist) if dist else None
        elif typ==3: # M/M/∞
            r_= rho_i[i]
            from math import factorial
            dist=[]
            for k_ in range(maxk+1):
                val= np.exp(-r_)* (r_**k_)/ factorial(k_)
                dist.append(val)
            pi_i_k[i]= np.array(dist)
        else:
            pi_i_k[i]= None

    return {
        'e_i_r': e_i_r,
        'rho_i_r': rho_i_r,
        'rho_i': rho_i,
        'K_i_r': K_i_r,
        'K_i':   K_i,
        'pi_i_k': pi_i_k
    }


def run_bcmp_analysis():
    """Uruchamia analizę BCMP i rysuje wykresy obciążenia, e_{i,r}, rho_{i,r}."""
    cfg = CONFIG
    N=7
    R=3
    routing_4d = build_routing_matrix_4d(cfg)

    lam = cfg['arrival_lambda']
    pcls= cfg['class_probs']
    arr_rates = np.array([lam*pcls[0], lam*pcls[1], lam*pcls[2]])

    node_types= np.array([1,2,3,2,2,3,2])  
    servers = np.array([
        cfg['registration_servers'],
        cfg['admissions_servers'],
        cfg['gynecology_servers'],
        cfg['predelivery_servers'],
        cfg['delivery_servers'],
        cfg['icu_servers'],
        cfg['postpartum_servers']
    ],dtype=int)

    service_rates= np.zeros((N,R))
    service_rates[0,:]= cfg['registration_rate']
    service_rates[1,:]= cfg['admissions_rate']
    service_rates[2,:]= cfg['gynecology_rate']
    service_rates[3,:]= cfg['predelivery_rate']
    service_rates[4,:]= cfg['delivery_rate']
    service_rates[5,:]= cfg['icu_rate']
    service_rates[6,:]= cfg['postpartum_rate']

    result= compute_bcmp_open_network_4d(
        N, R,
        arrival_rates=arr_rates,
        service_rates=service_rates,
        node_types=node_types,
        servers_per_node=servers,
        routing_matrix_4d=routing_4d,
        max_k_for_print=10
    )

    node_labels= [
       "Rejestracja",
       "Izba przyjęć",
       "O.gienekologiczny",
       "S.przedporodowa",
       "S.porodowa",
       "OIOM",
       "S.poporodowa"
    ]

    # --- Obciążenie węzłów ---
    rho_i= result['rho_i']
    plt.figure(figsize=(12,5))
    plt.bar(range(N), rho_i, color='green')
    plt.xticks(range(N),node_labels,rotation=0)
    plt.ylabel("Obciążenie ρ_i")
    plt.title("Obciążenie węzłów")
    plt.grid(True)
    plt.show()

    # --- e_{i,r} ---
    e_i_r= result['e_i_r']
    class_names= ["Bez komplikacji.","Z komplikacjami","Krytyczne"]
    class_colors= ["green","orange","red"]

    plt.figure(figsize=(12,5))
    index= np.arange(N)
    bar_width= 0.25
    for r_ in range(R):
        x_= index + (r_-(R-1)/2)*bar_width
        plt.bar(x_, e_i_r[:,r_], bar_width,
                color=class_colors[r_],
                label=class_names[r_] if r_==0 else None)
    plt.legend(class_names)
    plt.xticks(index,node_labels,rotation=0)
    # plt.title("Prawdopodobieństwo odwiedzenia węzła e_{i,r}")
    plt.title("Prawdopodobieństwo odwiedzenia węzła")
    plt.ylabel("Prawdopodobieństwo")
    plt.grid(True,axis='y')
    plt.show()

    # --- rho_{i,r} ---
    rho_i_r= result['rho_i_r']
    plt.figure(figsize=(12,5))
    for r_ in range(R):
        x_= index + (r_-(R-1)/2)*bar_width
        plt.bar(x_, rho_i_r[:,r_], bar_width,
                color=class_colors[r_],
                label=class_names[r_] if r_==0 else None)
    plt.legend(class_names)
    plt.xticks(index,node_labels,rotation=0)
    # plt.title("Obciążenia cząstkowe ρ_{i,r}")
    plt.title("Obciążenia cząstkowe")
    plt.grid(True,axis='y')
    plt.show()

    return result