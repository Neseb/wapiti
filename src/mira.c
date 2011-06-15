/*
 *      Wapiti - A linear-chain CRF tool
 *
 * Copyright (c) 2009-2011  CNRS
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "options.h"
#include "progress.h"
#include "sequence.h"
#include "tools.h"
#include "vmath.h"

#include "decoder.h"
#include "trainers.h"
#include "fmesure.h"

//Algorithme MIRA**/

void trn_mira(mdl_t *mdl) {
	const size_t  Y = mdl->nlbl;
	const size_t  F = mdl->nftr;
	//	const int     U = mdl->reader->nuni;
	//	const int     B = mdl->reader->nbi;
	const int     S = mdl->train->nseq;
	const int     K = mdl->opt->maxiter;
const double C = 1;
//const double C = mdl->opt->mira.C;
//	const double alpha = mdl->opt->mira.alpha;
	double 	*w = mdl->theta;	
	//wsum : somme de tous les poids
	//TODO : vectoriser tout ça
	double* wsum = xmalloc(F*sizeof(double));
	for(size_t i = 0 ; i < F ; i++)
		wsum[i] = w[i];
	//Nombre total de mises à jour de w
	int N = 0;


	// We will process sequences in random order in each iteration, so we
	// will have to permute them. The current permutation is stored in a
	// vector called <perm> shuffled at the start of each iteration. We
	// just initialize it with the identity permutation.
	int *perm = xmalloc(sizeof(int) * S);
	for (int s = 0; s < S; s++)
		perm[s] = s;

	for (int k = 0 ; k < K && !uit_stop; k++) { 
		//pour un nombre maxiter de fois

		// First we shuffle the sequence by making a lot of random swap
		// of entry in the permutation index.
		for (int s = 0; s < S; s++) {
			const int a = rand() % S;
			const int b = rand() % S;
			const int t = perm[a];
			perm[a] = perm[b];
			perm[b] = t;
		}
		// And so, we can process sequence in a random order
		for (int sp = 0; sp < S && !uit_stop; sp++) {
			const int s = perm[sp];
			const seq_t *seq = mdl->train->seq[s];
			int T = seq->len;
			size_t* out = xmalloc(T * sizeof(size_t)); //tableau 
			//de labels
			tag_viterbi(mdl, seq, out,NULL,NULL);
			bool differents = false;
			//On commence par regarder si le meilleur (out) est 
			//la référence (seq)
			int featCount = 0;
			double featSum = 0;
			for(int t = 0 ; t < T ; t++) {
				//Pour chaque unité dans les séquences : 
				//(les deux en ont autant)
				if (out[t] != (seq->pos[t]).lbl ) { 
					//si les deux labels sont différents
					differents = true;
				// Norme de \delta h : pour chaque, si égal 0 si différents +1
				// = count
					featCount++;	
				}
			}
			if (differents) { 
				// si  y != y(s) (le meilleur n'est pas 
				// la référence)
				// theta = theta + marge = theta + 
				// alpha*(features(y(t),x(t))- features(y,x(s)))
				//Pour chaque position t dans y, pour 
				//chaque feature activée par (y,s,x) :
				// On prend en compte les features 
				// unigrammes du premier mot

				//Alpha dépend de ....
				//On a alpha = max(0,min(C,(loss(out,seq) 
				// - \delta H_(t-1))/norme(\delta h_(t-1))

				//On calcule alpha
				//L = 1 - fmesure(out, pos, T)
				// \delta H : pour chaque si égal 0 si différents +(différence
				// des poids)

				// Pour le premier mot on ne regarde que les feat. unigrammes
				const pos_t* pos = &(seq->pos[0]);
				size_t y = out[0]; 
				size_t yt = pos->lbl;
				for(size_t p = 0 ; p < pos->ucnt && !uit_stop ; p++) {
					const size_t o = pos->uobs[p];
					featSum += w[mdl->uoff[o] + yt] - w[mdl->uoff[o] + y]; 
				}
				//Pour tous les mots suivants, on regarde 
				//à la fois les unigrammes et les bigrammes
				for(int t = 1 ; t < T  && !uit_stop ; t++) { 
					pos = &(seq->pos[t]);
					y = out[t]; 
					yt = pos->lbl;
					size_t yp = out[t-1]; 
					size_t ypt = seq->pos[t-1].lbl; 
					for(size_t p = 0 ; p < pos->ucnt ; p++) {
						const size_t o = pos->uobs[p];
					featSum += w[mdl->uoff[o] + yt] - w[mdl->uoff[o] + y]; 
					}
					for(size_t p = 0 ; p < pos->bcnt  && !uit_stop ; p++) {
						const size_t o = pos->bobs[p];
						size_t d = Y*yp + y;
						size_t dt = Y*ypt + yt;
					featSum += w[mdl->boff[o] + dt] - w[mdl->boff[o] + d];
					} 
				}
				double L = (1 - fmesure(out,seq,Y) - featSum) / (double) featCount;
				double alpha = (L < C) ? ((L > 0) ? L : 0) : C ;
			// Maintenant qu'on a calculé alpha, on peut appliquer l'update perceptron
				pos = &(seq->pos[0]);
				y = out[0]; 
				yt = pos->lbl;
				for(size_t p = 0 ; p < pos->ucnt && !uit_stop ; p++) {
					const size_t o = pos->uobs[p];
					w[mdl->uoff[o] + yt] += alpha;
					w[mdl->uoff[o] + y] -= alpha;
					for(size_t i = 0 ; i < F ; i++)
						wsum[i] += w[i];
					N++;
				}
				//Pour tous les mots suivants, on regarde 
				//à la fois les unigrammes et les bigrammes
				for(int t = 1 ; t < T  && !uit_stop ; t++) { 
					const pos_t *pos = &(seq->pos[t]);
					size_t y = out[t]; 
					size_t yt = pos->lbl;
					size_t yp = out[t-1]; 
					size_t ypt = seq->pos[t-1].lbl; 
					for(size_t p = 0 ; p < pos->ucnt ; p++) {
						const size_t o = pos->uobs[p];
						w[mdl->uoff[o] + yt] += alpha;
						w[mdl->uoff[o] + y] -= alpha;
						for(size_t i = 0 ; i < F ; i++)
							wsum[i] += w[i];
						N++;
					}
					for(size_t p = 0 ; p < pos->bcnt  && !uit_stop ; p++) {
						const size_t o = pos->bobs[p];
						size_t d = Y*yp + y;
						size_t dt = Y*ypt + yt;
						w[mdl->boff[o] + dt] += alpha;
						w[mdl->boff[o] + d] -= alpha;
						for(size_t i = 0 ; i < F ; i++)
							wsum[i] += w[i];
						N++;
					} 
				}
			}
			free(out);
		}
		//TODO : rajouter des uit_stop
		for(size_t i = 0 ; i < F ; i++)
			w[i] = wsum[i] / N;
		// Repport progress back to the user
		if (!uit_progress(mdl, k + 1, -1.0))
			break;
	}
	free(wsum);
	free(perm);
};
/*
theta : contient toutes les features présentes dans le train (tous les tests sur les observations, * Y feat unigrammes * YY feat bigrammes)
qui??? génère les feat pour une phrase (d'entrainement...) donnée ?
-> défini dans /sequence.h/, |uobs| est créé dans _dotrain_

l'algo d'entraînement est appelé dans dotrain(mdl_t *mdl), avec comme argument un modèle (où les données d'entraînement ont été chargées : on a les uobs , les bobs dans seq->pos[i] : données de la séquence d'entraînement à la position $i$, cf sequence.h


// Pour chaque position dans la séquence
for (int t = 0; t < T; t++) {
// On récupère les données à cette position
const pos_t *pos = &(seq->pos[t]);
// Pour chaque label possible
for (size_t y = 0; y < Y; y++) {
double sum = 0.0;
for (size_t n = 0; n < pos->ucnt; n++) {
const size_t o = pos->uobs[n];
sum += x[mdl->uoff[o] + y];
}
for (size_t yp = 0; yp < Y; yp++)
(*psi)[t][yp][y] = sum;
}
}

*/
