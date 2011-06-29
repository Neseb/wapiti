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
#include <time.h>

#include "model.h"
#include "options.h"
#include "progress.h"
#include "sequence.h"
#include "tools.h"
#include "vmath.h"

#include "decoder.h"
#include "trainers.h"
#include "fmesure.h"

//Algorithme MIRA

void trn_mira(mdl_t *mdl) {
	//printf("point 0");
	/* initialize random seed: */
	srand ( time(NULL) );
	//printf("point 42");
	const size_t  Y = mdl->nlbl;
	const size_t  F = mdl->nftr;
	const int     S = mdl->train->nseq;
	const int     K = mdl->opt->maxiter;
	const double C = mdl->opt->mira.C;

	double 	*w = mdl->theta;	

	//wSum : somme de tous les poids
	double* wSum = xmalloc(F*sizeof(double));
	// Nombre de fois que l'élément n'a pas été mis à jour
	size_t* wCache = xmalloc(F*sizeof(size_t));
	for(size_t i = 0 ; i < F ; i++) {
		wSum[i] = w[i];
		wCache[i] = 0;
	}
	//Nombre total de mises à jour de w, devrait être à terme K*S
	size_t wN = 1;
	//printf("Point 1");
	const int N = mdl->opt->nbest;
	double alphaSum;			
	size_t featCount[N];
	double featSum[N];
	double delta[N];
	double alpha[N];

	// We will process sequences in random order in each iteration, so we
	// will have to permute them. The current permutation is stored in a
	// vector called <perm> shuffled at the start of each iteration. We
	// just initialize it with the identity permutation.
	int *perm = xmalloc(sizeof(int) * S);
	for (int s = 0; s < S; s++)
		perm[s] = s;

	for (int k = 0 ; k < K && !uit_stop; k++) { 
		//pour un nombre maxiter de fois

		/*		// First we shuffle the sequence by making a lot of random swap
		// of entry in the permutation index.
		for (int s = 0; s < S; s++) {
		const int a = rand() % S;
		const int b = rand() % S;
		const int t = perm[a];
		perm[a] = perm[b];
		perm[b] = t;
		}
		*/
		// And so, we can process sequence in a random order
		for (int sp = 0; sp < S && !uit_stop; sp++) {
			const int s = perm[sp];
			const seq_t *seq = mdl->train->seq[s];
			int T = seq->len;
			size_t* out = xmalloc((size_t) T * N * sizeof(size_t)); //tableau de labels
			size_t (*out_2d)[T][N] = (void *) out; 
			// On récupère les n-best
			tag_nbviterbi(mdl, seq, N, *out_2d,NULL,NULL);

			//printf("On itère sur les n-best pour calculer les deltas correspondant à chacun");
			for (int n =0; n < N; n++) {
				printf("\n sp = %d ",sp);
				featCount[n] = 0;
				featSum[n] = 0;
				printf("featSum = %g ",featSum[n]);
				delta[n] = 0;
				alpha[n] = 0;

				//On commence par regarder  si l'hypothèse  (out[][n]) est 
				//la référence (seq)
				// featCount = \| h(e^t,f^t)-h(e,f^t)\|² = cardinal de la différence 
				// symétrique entre les caracts de la ref et celle de l'hyp
				for(int t = 0 ; t < T ; t++) {
					//Pour chaque position dans les séquences : 
					//(les deux en ont autant)
					if ((*out_2d)[t][n] != (seq->pos[t]).lbl ) { 
						//si les deux labels sont différents
						// Norme de \delta h : pour chaque position, si égal 0 si différents +1
						featCount[n]++;
					}
				}
				printf("featCount = %lu ", featCount[n]);
				if (featCount[n]) { 
					//On calcule delta pour chaque hypothèse en cours
					//L = 1 - fmesure(out, pos, T)
					// \delta H : pour chaque si égal 0 si différents +(différence
					// des poids)

					// Pour le premier mot on ne regarde que les feat. unigrammes
					const pos_t* pos = &(seq->pos[0]);
					size_t y = (*out_2d)[0][n];
					size_t ys = pos->lbl;
					for(size_t p = 0 ; p < pos->ucnt ; p++) {
						const size_t o = pos->uobs[p];
						featSum[n] += w[mdl->uoff[o] + ys] - w[mdl->uoff[o] + y]; 
					}
					//Pour tous les mots suivants, on regarde 
					//à la fois les unigrammes et les bigrammes
					for(int t = 1 ; t < T ; t++) { 
						pos = &(seq->pos[t]);
						y = (*out_2d)[t][n]; 
						ys = pos->lbl;
						size_t yp = (*out_2d)[t-1][n]; 
						size_t yps = seq->pos[t-1].lbl; 
						for(size_t p = 0 ; p < pos->ucnt ; p++) {
							const size_t o = pos->uobs[p];
							featSum[n] += w[mdl->uoff[o] + ys] - w[mdl->uoff[o] + y]; 
						}
						for(size_t p = 0 ; p < pos->bcnt  && !uit_stop ; p++) {
							const size_t o = pos->bobs[p];
							size_t d = Y*yp + y;
							size_t ds = Y*yps + ys;
							featSum[n] += w[mdl->boff[o] + ds] - w[mdl->boff[o] + d];
						} 
					}
					delta[n] = (1 - nfmesure(N,n,*out_2d,seq,Y) - featSum[n]) / (double) featCount[n];

				} ;
				printf("featsum = %f ", featSum[n]);	
				printf("delta = %f ", delta[n]);					
			};	

			//printf(" On calcule les \alpha en faisant en sorte que la somme reste inférieure à C");
			alphaSum = 0;			
			int rough[N]; 
			for(int n =0;n<N;n++) rough[n] = 0;
			int nTmp = 0;	
			while(nTmp < N) { 
				int n = rand() % N;
				if(!rough[n]) {
					if(delta[n] > 0) {
						double b = delta[n];
						if(alphaSum + b > C) { 
							alpha[n] += C-alphaSum;
							break;
						}
						alpha[n] += b;
						printf("Somme : %f ",alphaSum);
					}
					rough[n] = 1;
					nTmp++;
				}
			};
			
for(int n= 0;n< N;n++) printf("alpha(%d) = %g, alphaSum = %g ", n, alpha[n],alphaSum);

			/*			while(alphaSum < C) { 
						int n = rand() % N;
			//	printf("%d\n",n); 
			//	printf("\\delta(n) : %f\n",delta[n]);
			if(delta[n] > 0) {
			alpha[n] += delta[n];
			alphaSum += delta[n];
			printf("Somme : %f\n",alphaSum);
			}
			}*/

			//printf("On itère sur les N-best pour faire l'update perceptron " );
			for(int n=0; n < N; n++) {
				if (featCount[n]) { 
					// Maintenant qu'on a calculé les  alpha, on peut appliquer l'update perceptron
					// Pour le premier mot on ne regarde que les feat. unigrammes
					const pos_t* pos = &(seq->pos[0]);
					const size_t y = (*out_2d)[0][n]; 
					const size_t ys = pos->lbl;
					for(size_t p = 0 ; p < pos->ucnt ; p++) {
						const size_t o = pos->uobs[p];
						const size_t j = mdl->uoff[o] + y;		
						const size_t js = mdl->uoff[o] + ys;		
						wSum[j] += (wN - wCache[j]) * w[j];
						wSum[js] += (wN - wCache[js]) * w[js];
						w[js] += alpha[n];
						w[j] -= alpha[n];
						wCache[js] = wN;
						wCache[j] = wN;
					}
					//Pour tous les mots suivants, on regarde 
					//à la fois les unigrammes et les bigrammes
					for(int t = 1 ; t < T ; t++) { 
						const pos_t *pos = &(seq->pos[t]);
						const size_t y = (*out_2d)[t][n]; 
						const size_t ys = pos->lbl;
						const size_t yp = (*out_2d)[t-1][n]; 
						const size_t yps = seq->pos[t-1].lbl; 
						for(size_t p = 0 ; p < pos->ucnt ; p++) {
							const size_t o = pos->uobs[p];
							const size_t j = mdl->uoff[o] + y;		
							const size_t js = mdl->uoff[o] + ys;		
							wSum[j] += (wN - wCache[j]) * w[j];
							wSum[js] += (wN - wCache[js]) * w[js];
							w[js] += alpha[n];
							w[j] -= alpha[n];
							wCache[js] = wN;
							wCache[j] = wN;
						}
						for(size_t p = 0 ; p < pos->bcnt ; p++) {
							const size_t o = pos->bobs[p];
							const size_t j = mdl->boff[o] + Y*yp + y;		
							const size_t js = mdl->boff[o] + Y*yps + ys;		
							wSum[j] += (wN - wCache[j]) * w[j];
							wSum[js] += (wN - wCache[js]) * w[js];
							w[js] += alpha[n];
							w[j] -= alpha[n];
							wCache[js] = wN;
							wCache[j] = wN;
						} 		
					}
				}
				wN++;
			} 
			free(out);
		}		
		for(size_t i = 0 ; i < F ; i++)
			w[i] =  (wSum[i] + (wN - wCache[i]) * w[i]) / (double) wN;
		// Repport progress back to the user
		if (!uit_progress(mdl, k + 1, -1.0))
			break;
	}
	free(wSum);
	free(wCache);
	free(perm);
};
