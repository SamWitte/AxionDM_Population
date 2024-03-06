function getDM(r, npts)
    RSun = 8300 # pc
    delR = r .- [0.0 RSun 0.0]
    xvals = LinRange(0.0, delR[1], npts)
    yvals = LinRange(RSun, delR[2] .+ RSun, npts)
    zvals = LinRange(0.0, delR[3], npts)
    dr =  sqrt.((xvals[2] - xvals[1]).^2 .+ (yvals[2] - yvals[1]).^2 .+ (zvals[2] - zvals[1]).^2)
    DM = 0.0
    for i in 1:npts
        DM += YMW_ne([xvals[i] yvals[i] zvals[i]]) .* dr
    end
    return DM
end

function YMW_ne(r)
    x,y,z = r

    RSun = 8300 # pc
    
    R = sqrt.(x.^2 .+ y.^2)
    phi = atan(y, x)
    R_w = 8400.0
    gamma_w = 0.14
    z_c = gamma_w .* (R .- R_w)
    z_w = z_c .* cos.(phi .- 0.0)
    
    
    Bd = 15000.0
    Ad = 2500
    H1 = 1673.0
    
    # thick disk
    if R < Bd
        g_d = 1.0
    else
        g_d = sech.( (R - Bd) ./ Ad).^2
    end
    n1_0 = 0.01132 # cm-3
    n1 = n1_0 .* g_d .* sech.( (z - z_w) ./ H1).^2
    
    # check if in FB
    bFB = RSun * tan.(20  * pi / 180)
    aFB = 0.5 * RSun * tan.(50  * pi / 180)
    zFB = 0.5 .* RSun .* tan.(50 .* pi / 180.0)
    PFB = (x ./ bFB).^2 .+ (y ./ bFB).^2 .+ ((z .- zFB) ./ aFB).^2
    PFB2 = (x ./ bFB).^2 .+ (y ./ bFB).^2 .+ ((z .+ zFB) ./ aFB).^2
    in_FB = false
    if (PFB < 1) || (PFB2 < 1)
        in_FB = true
        JFB = 1.0
        n1 *= JFB
    end
    
    # thin disk
    B_2 = 4000 # pc
    A_2 = 1200 # pc
    n2_0 = 0.404
    H = 32 .+ 1.6 .* 1e-3 .* R .+ 4e-7 .* R.^2 # R [pc]
    K2 = 1.54
    n2 = n2_0 * g_d .* sech.( (R - B_2) ./ A_2).^2 .* sech.( (z - z_w) ./ (K2 .* H)).^2
    
    
    # spiral arms
    na = 0.0
    Rai = [3.35 3.71 3.56 3.67 8.21] .* 1e3 # pc
    phi_ai = [44.4 120.0 218.6 330.3 55.1] .* pi / 180.0 # rad
    psi_ai = [11.43 9.84 10.38 10.54 2.77].* pi / 180.0  # rad
    n_ai = [0.135 0.129 0.103 0.116 0.0057]
    w_ai = [300 500 300 500 300]
    nCN = 2.40
    phiCN = 109 .* pi / 180
    delPhiCN = 8.2 .* pi / 180
    nSG = 0.626
    phiSG = 75.8 .* pi / 180
    delPhiSG = 20.0 .* pi / 180
    
    Ka = 5.01
    A_a = 11680.0
    
    # cspitch = [0.98 0.985 0.9836 0.983 0.9988]
    
    for i in 1:5
        
        # armr = Rai[i] .* exp.((phi .+ 2 * pi .- phi_ai[i]) .* tan.(psi_ai[i]))
        if phi < 0
            phi_mod = phi + 2*pi
        else
            phi_mod = phi
        end
        
        armr = Rai[i] .* exp.((phi_mod .- phi_ai[i]) .* tan.(psi_ai[i]))
        # s_ai = abs.(R .- armr);
        phi_sa = log.(R ./ Rai[i]) ./ tan.(psi_ai[i]) .+ phi_ai[i]
        s_ai = R .* sqrt.((cos.(phi) .- cos.(phi_sa)).^2 .+ (sin.(phi) .- sin.(phi_sa)).^2)
        
        efac = 1.0
        if i == 3
            if phi < phiCN
                efac = (1 .+ nCN .* exp.(- ((phi_ai[i] .- phiCN)./ delPhiCN).^2)) .* (1 .- nSG .* exp.(- ((phi_ai[i] .- phiSG)./ delPhiSG).^2))
            else
                efac = (1 .+ nCN) .* (1 .- nSG .* exp.(- ((phi_ai[i] .- phiSG)./ delPhiSG).^2))
            end
        end
        na += g_d .* n_ai[i] .* efac .* sech.(s_ai ./ w_ai[i]).^2 .* sech.((R .- B_2) ./ A_a).^2 .* sech.((z .- z_w) ./ (Ka .* H)).^2
    end
    
    # GC
    xGC = 50 #pc
    yGC = 0.0
    zGC = -7.0
    AGC = 160.0
    H_GC = 35.0
    nGC_0 = 6.2
    RGC = sqrt.((x .- xGC).^2 .+ (y .- yGC).^2)
    nGC = nGC_0 .* exp.(- (RGC ./ AGC).^2) .* sech.( (z .- zGC)./H_GC).^2
    
    # Gum nebula
    xGN = -446.0
    yGN = RSun + 47
    zGN = -31.0
    nGN0 = 1.84
    WGN = 15.1
    AGN = 125.8
    KGN = 1.4
    GumTest = ((x .- xGN) ./ AGN).^2 .+ ((y .- yGN) ./ AGN).^2 .+ ((z .- zGN) ./ (KGN .* AGN)).^2
    if GumTest < 1
    
        theta = abs.(atan((z .- zGN)/sqrt((x .- xGN).^2 .+ (y .- yGN).^2)));
        zp = AGN.^2 .* KGN ./ sqrt.(AGN.^2 .+ AGN.^2 .* KGN.^2 ./ tan.(theta).^2)
        xyp = zp ./ tan.(theta);
        
        rp=sqrt.(zp.^2 .+ xyp.^2);
        if((AGN .- abs.(xyp))<1e-15)
            alpha=PI/2;
        else
            alpha=-atan((-(AGN)*(KGN)*xyp)/((AGN)*sqrt((AGN)*(AGN)-xyp*xyp)));
        end
        rr  = sqrt.((x .- xGN).^2 .+ (y .- yGN).^2 .+ (z .- zGN).^2)
        rp = sqrt.(zp.^2 .+ xyp.^2)
        sGN = abs.((rr .- rp) .* sin.(theta .+ alpha));
        
        nGN = nGN0 .* exp( - (sGN ./ WGN).^2)
        GN = true
    else
        nGN = 0.0
        GN = false
    end
    
    
    # local bubble
    JLB = 0.480
    RLB = 110.0
    rLB = sqrt.(x.^2 .+ (0.94 .* (y .- RSun .- 40.0) .- 0.34 .* z).^2)
    in_LB = false
    if rLB < RLB
        in_LB = true
  
        nLB1_0 = 1.094
        thetaLBI = 195.4 * pi / 180
        dThetaLBI = 28.4 * pi / 180
        W_LB1 = 14.2
        HLB1 = 112.9
        nLB2_0 = 2.33
        thetaLB2 = 278.2 * pi / 180
        dThetaLB2 = 14.7 * pi / 180
        W_LB2 = 15.6
        HLB2 = 43.6
        
        
        nLB1 = nLB1_0 .* sech.((phi - thetaLBI) ./ dThetaLBI).^2 .* sech.((rLB - RLB) ./ W_LB1).^2 .* sech.((z) ./ HLB1).^2
        nLB2 = nLB2_0 .* sech.((phi - thetaLB2) ./ dThetaLB2).^2 .* sech.((rLB - RLB) ./ W_LB2).^2 .* sech.((z) ./ HLB2).^2
    else
        nLB1 = 0.0
        nLB2 = 0.0
    end
    
    # Loop I
    xLI = -10.0
    yLI = 8106.0
    zLI = 10.0
    rLI = sqrt.( (x .- xLI).^2 .+ (y .- yLI).^2 .+ (z .- zLI).^2)
    nLI_0 = 1.907
    RLI = 80.0
    WLI = 15.0
    theta_LI = 40.0 * pi / 180
    dtheta_LI = 30.0 * pi / 180
    nLI = nLI_0 .* exp.(- ((rLI .- RLI) ./ WLI).^2) .* exp.( - (theta_LI ./ dtheta_LI).^2)
    if rLI < RLI
        LoopI = true
    else
        LoopI = false
    end
    
    if in_FB
        n0 = JFB .* n1 .+ maximum([n2, na])
    end
    
    wLB = 0.0
    if in_LB
        n0 = JLB .* n1 .+ maximum([n2, na])
        if ((nLB1 .+ nLB2) > n0)&&((nLB1 .+ nLB2) > nGN)
            wLB = 1.0
        end
    else
        n0 = n1 .+ maximum([n2, na])
    end
    
    wLI = 0.0
    if LoopI
        if nLI > n0
            wLI = 1.0
        end
    end
    
    wGN = 0.0
    if GN
        if nGN > n0
            wGN = 1.0
        end
    end

    
    nGal = (1 .- wLB) .* ((1 .- wGN) .* ((1 .- wLI) .* (n0 .+ nGC) .+ wLI .* nLI) .+ wGN .* nGN) .+ wLB .* (nLB1 .+ nLB2)
    return nGal
end
