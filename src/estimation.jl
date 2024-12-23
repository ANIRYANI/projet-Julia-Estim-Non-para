module Estimation

using ..Kernels
using LinearAlgebra  # résolution des systèmes linéaires

# Estimation de la fonction de répartition cumulative (FRC)
"""

Estime la fonction de répartition cumulative (FRC) en un point x pour un ensemble de données `.

# Arguments
- data : Un vecteur des données.
- x : Le point où on veut estimer la CDF.

# Retourne
La valeur de la FRC en x.
"""
function FRC(data::Vector{Float64}, x::Float64)
    n = length(data)
    return sum(x_i <= x for x_i in data) / n
end

# Estimation de la densité par noyau 
"""

Estime la densité en un point x en utilisant l'estimation de densité classique.

# Arguments
- data : Un vecteur des données.
- x : Le point où la densité est estimée.
- h : largeur de bande.

# Retourne
La valeur estimée de la densité en x.
"""
function densite_classique(data::Vector{Float64}, x::Float64, h::Float64)
    n = length(X)
    num = sum(Y[i] * rectang_kernel((x - X[i]) / h) for i in 1:n)
    den = sum(rectang_kernel((x - X[i]) / h) for i in 1:n)
    return den > 0 ? num / den : NaN
end


# Estimation de la fonction de survie avec Kaplan-Meier
"""

Estime la fonction de survie en utilisant l'estimateur de Kaplan-Meier.

# Arguments
- temps : Un vecteur des temps d'observation (peut inclure des censures).
- events : Un vecteur booléen où `true` indique un événement observé (par ex. décès), et `false` une censure.

# Retourne
Un tuple contenant deux vecteurs : les temps uniques triés et les probabilités de survie correspondantes.
"""
function survie_KM(temps::Vector{Float64}, events::Vector{Bool})
    # Trier les temps et événements selon les temps croissants
    indices = sortperm(temps)
    temps = temps[indices]
    events = events[indices]

    # Identifier les temps uniques et compter les événements
    temps_uniques = unique(temps)
    n_risque = length(temps)  # Nombre de sujets en risque
    survie = Float64[]        # Vecteur des probabilités de survie

    s = 1.0  # Probabilité initiale (1 au début)
    push!(survie, s)

    for t in temps_uniques
        # Nombre d'événements et de censures à ce temps
        d = sum((temps .== t) .& evenements)  # Décès (événements observés)
        n_risque -= sum(temps .== t)         # Réduction du nombre en risque

        # Mise à jour de la probabilité de survie
        s *= 1 - d / (n_risque + d)  # Kaplan-Meier
        push!(survie, s)
    end

    # Enlever la probabilité initiale pour synchroniser les indices avec les temps
    return (temps_uniques, survie[2:end])
end


# Estimation de Nadaraya-Watson
"""

Estime la valeur attendue de Y  pour une valeur x donnée de  X  en utilisant l'estimateur de Nadaraya-Watson.

# Arguments
- X : Vecteur des prédicteurs.
- Y : Vecteur des réponses.
- x : La valeur pour laquelle l'estimation est calculée.
- h : largeur de bande.
- noyau : La fonction de noyau.

# Retourne
La valeur estimée en x.
"""
function nadaraya_watson(X::Vector{Float64}, Y::Vector{Float64}, x::Float64, h::Float64, noyau::Function)
    n = length(X)
    num = sum(Y[i] * noyau((x - X[i]) / h) for i in 1:n)
    den = sum(noyau((x - X[i]) / h) for i in 1:n)
    return den > 0 ? num / den : NaN
end


# Estimation par polynômes locaux
"""
Estime la fonction de régression en un point x en utilisant un estimateur par polynômes locaux.

# Arguments
- X : Vecteur des prédicteurs.
- Y : Vecteur des réponses.
- x : Le point où l'estimation est calculée.
- h : Largeur de bande .
- p : Degré du polynôme local.
- noyau : Fonction de noyau.

# Retourne
La valeur estimée de la fonction de régression en x.
"""
function polynomes_locaux(X::Vector{Float64}, Y::Vector{Float64}, x::Float64, h::Float64, p::Int, noyau::Function)
    n = length(X)

    # Construction des poids du noyau
    W = Diagonal([noyau((x - X[i]) / h) for i in 1:n])

    # Matrice de Vandermonde pour le polynôme de degré p
    Z = [X .^ j for j in 0:p] |> hcat  # Crée une matrice où chaque colonne est X^j

    # Ajustement par moindres carrés pondérés
    beta = (Z' * W * Z) \ (Z' * W * Y)

    # Le premier coefficient est l'estimation en x (terme constant)
    return beta[1]
end


# Estimation par splines cubiques
"""

Estime la fonction de régression par la méthode des splines cubiques.

# Arguments
- X : Vecteur des prédicteurs (variable indépendante), trié par ordre croissant.
- Y : Vecteur des réponses (variable dépendante).
- x : Le point où l'estimation est effectuée.

# Retourne
La valeur estimée de la fonction de régression en x.
"""
function splines_cubiques(X::Vector{Float64}, Y::Vector{Float64}, x::Float64)
    n = length(X)

    # la matrice des différences h
    h = diff(X)

    # Construction de la matrice tridiagonale
    M = zeros(n, n)
    M[1, 1] = 1
    M[n, n] = 1
    for i in 2:n-1
        M[i, i-1] = h[i-1]
        M[i, i] = 2 * (h[i-1] + h[i])
        M[i, i+1] = h[i]
    end

    # Construction du vecteur des différences de Y
    d = zeros(n)
    for i in 2:n-1
        d[i] = 3 * ((Y[i+1] - Y[i]) / h[i] - (Y[i] - Y[i-1]) / h[i-1])
    end

    # obtenir les coefficients c
    c = M \ d

    # Calcul des coefficients b et d
    b = zeros(n-1)
    d = zeros(n-1)
    for i in 1:n-1
        b[i] = (Y[i+1] - Y[i]) / h[i] - h[i] * (2 * c[i] + c[i+1]) / 3
        d[i] = (c[i+1] - c[i]) / (3 * h[i])
    end

    # Évaluation de la spline
    i = searchsortedfirst(X, x) - 1
    i = max(min(i, n-1), 1)  # Assurer que i est dans les limites
    dx = x - X[i]
    return Y[i] + b[i] * dx + c[i] * dx^2 + d[i] * dx^3
end


####
export FRC, densite_noyau, survie_KM, nadaraya_watson, polynomes_locaux, splines_cubiques


end
