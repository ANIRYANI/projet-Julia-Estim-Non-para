# Import necessary packages
using NonParametricEstimation
using NonParametricEstimation.Estimation
using Plots  
using Random

###----------------------------------Exemple: Estimation de la fonction de répartition--------------------------------------------
# Générer un échantillon
n = 1000  # Nombre  d'observations
lambda = 1.0  # paramètre loi exponentielle
data = rand(Exponential(lambda), n)

# grille de valeurs x
x_vals = 0:0.1:5  #

# Estimation de la fonction de répartition  
cdf_estimates = [FRC(data, x) for x in x_vals]
# la vraie courbe
true_cdf = cdf -> 1 - exp(-lambda * cdf)  

# dessiner les deux courbes
plot(x_vals, cdf_estimates, label="CDF estimée", linewidth=2, linestyle=:dash)
plot!(x_vals, true_cdf.(x_vals), label="Vraie CDF", linewidth=2)
xlabel!("x")
ylabel!("CDF")
title!("Estimatée vs vraie cdf de la fonction exponentielle")
legend!(:topright)

###----------------------------------Exemple: Estimation de la densité --------------------------------------------

# Etape 1: Générer des données 
n = 500  # Nombre d'observations
Random.seed!(123) 
x_vals = sort(rand(n)) 
# Fonction de régression réelle : y = sin(2π * x) + bruit
true_func(x) = sin(2 * π * x)
bruit = 0.1 * randn(n)  # Ajout d'un bruit

# grille de valeurs x
x_vals = -4:0.1:4  
y_vals = true_func.(x_vals) .+ bruit  # Valeurs observées de y


# la régression avec l'estimateur de Nadaraya-Watson 
nw_reg = [nadaraya_watson(x_vals, y_vals, x, 0.1, gaussian_kernel) for x in x_vals]

# Régression polynomes locaux (degré 2)
lpoly_reg = [plynomes_locaux(x_vals, y_vals, x, 0.1, 2,  gaussian_kernel) for x in x_vals]

# Régression par splines
spline_reg = [splines_cubiques(x_vals, y_vals, x) for x in x_vals]


# Tracer les résultats
plot(x_vals, true_func.(x_vals), label="Fonction réelle", linewidth=2, linestyle=:solid, color=:black)
plot!(x_vals, nadaraya_watson_reg, label="Estimateur Nadaraya-Watson", linewidth=2, linestyle=:dash)
plot!(x_vals, local_poly_reg, label="Régression polynomiale locale", linewidth=2)
plot!(x_vals, spline_reg, label="Régression par splines", linewidth=2)

# Personnaliser le graphique
xlabel!("x")
ylabel!("y")
title!("Méthodes d'estimation de régression")
legend!(:topright)

###----------------------------------Exemple: Estimation de la densité --------------------------------------------
# Générer des données de survie simulées
n = 100  # Nombre d'observations
Random.seed!(123)  

# Générer des temps de survie simulés 
temps_de_survie = -log.(rand(n))  # temps de survie exponentiels avec un paramètre lambda = 1

# Générer un vecteur d'indicateurs de censure (0 = censuré, 1 = observé)
censure = rand(Bool, n)  

# Estimation de la fonction de survie avec la méthode de Kaplan-Meier
temps_surv_ordonnes, survie_estimee = survie_KM(temps_de_survie, censure)

# Etape 3: Tracer la fonction de survie estimée
plot(temps_de_survie_ordonnes, survie_estimee, label="Fonction de survie estimée (Kaplan-Meier)", linewidth=2)
xlabel!("Temps")
ylabel!("Probabilité de survie")
title!("Estimation de la fonction de survie de Kaplan-Meier")