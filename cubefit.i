#include "multiprofile.i"
#include "OptimPack1.i"
#include "cubeview.i"
#include "util_fr.i"

/*
   AUTHOR: Thibaut Paumard
           Damien Gratadour (l1l2, markov)
 */

// First save previous values of temporary function names
scratch = save(scratch,
               // methods
               cubefit_new, cubefit_view, cubefit_eval, cubefit_printer,
               cubefit_fit, cubefit_model,
               // static functions
               cubefit_l1l2, cubefit_markov, cubefit_corr,
               cubefit_op_f, cubefit_op_viewer, cubefit_op_printer
               );

extern cubefit;
func cubefit_new(args)
/* DOCUMENT cubefit class
            fitobj = cubefit(new, );

   Cubefit is an OXY class for designed for spectral fitting with
   spatial regularisation in a spectro-imaging context.

   The 3D model is based on a 1D model and 2D parameter maps. The 2D
   maps are regularised (using by default an L1L2 regularisation).

   The estimator is a compound of a chi^2 (based on the 1D model), a
   regularisation term (based of the 2D regularisation of the various
   2D parameter maps) and an optional decorrelation term (based on the
   crosscorrelation of specific pairs of paramter maps).
   
   MEMBERS
     cube:   the data to fit, a NX x NY x NZ array. The spectra are in
             the Z dimension.
     weight: optional (defaults to nil). An array with the same
             dimensions as CUBE, giving the relative weights of the
             data points for fitting. Set to 0. for points to ignore
             altogether. For Gaussian noise, this should be set to
             data root-mean-square (beware: this is the square root of
             what lmfit expects).
     fcn:    an LMFIT compatible 1D model function:
              model_1d = fcn(fcn_x, parameters, grad, deriv=1);
             model_1d must be an array(double, NZ).
     fcn_x:  whatever FCN accepts as its first positional
             argument. Often a wavelength axis, sometimes a complex
             object, possibly nil.
     regularisation: the regularisation function to use, by default
             cubefit.l1l2. Any function with the same prototype as
             l1l2 or markov is suitable. Set regularisation to nil
             remove regularisation entirely. Default: cubefit.markov.
     decorrelate: pairs of map ID for which the cross-correlation
             should ba minimal together with weight, e.g. decorrelate
             = [1, 2, 0.4] if the maps for parameters 1 and 2 should
             not be correlated, with a weight of 0.4 relative to the
             other terms of the estimator.
     scale, delta, pscale, poffset, ptweak: should be documented.

   METHODS
     These methods can be called as
         fitobj, METHOD, [parameters]
     or  result = fitobj(METHOD, [parameters]).

     The NEW method is a bit peculiar as it is normally called from
     the template CUBEFIT object itself:
         fitobj = cubefit(new, )
     It can however be called as
         fitobj2 = fitobj(new, )
     in which case the (potentially overriden) methods and static
     virtual functions are copied from FITOBJ to FITOBJ2. The members
     are not copied though (this may change in the future).

     The methods are all inherently "virtual" and can be overriden
     with
         fitobj, METHOD=new_implementation
   
     new:   create an instance of the class. This methods accepts a
            keyword for each of the cubefit members listed above.
     view:  visualise parameter maps
     eval:  compute the criterion (the function to minimise)
     model: compute a model cube from parameter maps (for visualisation)
     printer: print various things (in particular the criterion)
     fit:   performs the actual fit, using op_mnb()

   STATIC VIRTUAL FUNCTIONS
     Those functions are not "methods" as they don't "use" any
     member. They can be accessed as fitobj.FUNCTION() and can be
     overriden with fitobj, FUNCTION=newfunc.

     op_f:  wrapper around EVAL, suitable as the F argument of op_mnb()
     op_viewer: wrapper around view, suitable as VIEWER for op_mnb()
     op_printer: wrapper around printer, suitable as PRINTER for op_mnb()

   STATIC FUNCTIONS
     Those functions are in the cubefit object but not copied by NEW
     in the FIOBJ object. They can be called with: cubefit.FUNCTION().
   
     l1l2:   L1L2 regularisation method. e.g.:
               fitobj, regularisation=cubefit.l1l2
     markov: Markov regularisation method. e.g.:
               fitobj, regularisation=cubefit.markov

   SYNOPSIS
     fitobj = cubefit(new, [members=members])
     x = array(double, NX, NY, NPARMS); // initial guess
     fitobj, fit, x;
     fitobj, view, x;
   
   SEE ALSO: oxy, cubefit_METHOD, cubefit_FUNCTION, op_mnb, lmfit
 */
{
  local regularisation;
  local view, eval, printer, fit, model, op_f, op_viewer, op_printer, markov;
  restore, use, view, eval, printer, fit, model,
    op_f, op_viewer, op_printer, markov;
  
  // regularisation is treated specially: setting it to nil is
  // not the same as not specifying it.
  if (anyof(args(-)=="regularisation")) regularisation = args("regularisation");
  else regularisation=markov;
  
  return save(// Same for every cubefit object:
              // methods:
              view, eval, printer, fit, model,
              // static functions:
              op_f, op_viewer, op_printer,
              
              // Specific to this object (keywords):
              // function/methods
              fcn         = args("fcn"),
              fcn_x       = args("fcn_x"),
              regularisation,
              // data
              cube        = args("cube"),
              weight      = args("weight"),
              scale       = args("scale"),
              delta       = args("delta"),
              pscale      = args("pscale"),
              poffset     = args("poffset"),
              ptweak      = args("ptweak"),
              decorrelate = args("decorrelate")
              );
}
wrap_args, cubefit_new;

func cubefit_view(x, noscale=)
/* DOCUMENT fitobj, view, x

   View CUBEFIT patameter maps. Used internally for visualisation
   during a fit (see cubefit_fit), but can also be called direclty (in
   this case, NOSCALE should probably be set to 1).

   OBJECT
     FITOBJ    an instance of thecubefit class, created with cibefit.new
   
   PARAMETERS
     X         a stack of parameter maps
     
   KEYWORDS
     noscale=1 to ignore pscale and poffset. NOSCALE should almsot
               always be used when VIEW is used manually (in this
               case, X has not been rescaled consistently with pscale
               and poffset).

   SEE ALSO cubefit, cubefit_new
*/
{
  local pscale, poffset;
  restore, use, pscale, poffset;
  d = dimsof(x);

  if (noscale) {
    psc  = array(1., d(4));
    pof = array(0., d(4));
  } else {
    if (is_void(pscale)) {
      psc = array(1., d(4));
    } else {
      if (numberof(pscale)==1) psc = array(pscale, d(4));
      else psc = pscale;
    }
    if (is_void(poffset)) {
      pof = array(0., d(4));
    } else {
      if (numberof(poffset)==1) pof = array(poffset, d(4));
      else pof = poffset;
    }
  }
  
  for (k = 1; k <= d(4); ++k) {
    window, k-1; fma;
    pli, x(,,k);
    colorbar, min(x(,,k))*psc(k)+pof(k),max(x(,,k))*psc(k)+pof(k);
    pause, 1;
  }
}

func cubefit_model(x, noscale=)
/* DOCUMENT model_cube = fitobj(model, x)

     Create a model cube from a set of parameter maps. Like for the
     VIEW method, NOSCALE should almost always be set to 1.

   SEE ALSO: cubefit, cubefit_view
     
 */
{
////// WARNING: this function is for visualisation, it is "inlined" in eval
/////  for optimization
  use, fcn_x, fcn, pscale, poffset, ptweak;
  
  d = dimsof(x);
  if (noscale) {
    psc = array(1., d(4));
    pof = array(0., d(4));
  } else {
    psc = pscale;
    pof = poffset;
  }

  if (is_void(psc))   xs = x; else xs = x * psc(-,-,);
  if (!is_void(pof))  xs += pof(-,-,);
  if (!is_void(ptweak))  use_method, ptweak, xs, derivatives;
  
  nx = d(2); ny = d(3);
  nz = numberof(fcn(fcn_x, xs(1,1,)));

  y = array(double, nx, ny, nz);
  
  for (i = 1; i<=nx; ++i)
    for (j = 1; j<=ny; ++j)
      y(i,j,) = fcn(fcn_x, xs(i,j,));

  return y;
}

func cubefit_eval(x, &gx, noscale=, returnmaps=)
/* DOCUMENT cubefit.eval method
            criterion = fitobj(eval, x, gx)
         or maps = fitobj(eval, x, noscale=1, returnmaps=1)

   Like for the VIEW method, NOSCALE should almost always be set to 1
   when calling this method manually.

   Criterion is the sum of a weighted chi2 and a regularisation term.
   To remove regularisation altogether, set regularisation to nil.

   If RETURNMAPS is one, returns a stack of maps for each component of
   the criterion (chi2 term last) instead of the integral.

   SEE ALSO: cubefit, cubefit.l1l2, cubefit.markov,
             cubefit.model, cubefit.fit
 */
{
  // beware: "model" below is a local variable, not the method.
  local scale, delta, cube, fcn_x, fcn, weight, regularisation,
    pscale, poffset, ptweak, decorrelate;
  restore, use, scale, delta, cube, fcn_x, fcn, weight, regularisation,
    pscale, poffset, ptweak, decorrelate;

  d = dimsof(x);
  nx = d(2); ny = d(3);
  res = 0.;
  gx = array(double, d);

  if (noscale) {
    x = x; // copy input in order to not modify it!
    if (!is_void(psoffset)) x(..,) -= poffset(-,-,);
    if (!is_void(pscale))   x(..,) /= pscale(-,-,);
  }
  
  if (is_void(pscale))   xs = x; else xs = x * pscale(-,-,);
  if (!is_void(poffset)) xs += poffset(-,-,);
  if (!is_void(ptweak))  use_method, ptweak, xs, derivatives;
  if (!is_void(derivatives) && anyof(dimsof(derivatives) != dimsof(xs)))
    error("ptweak derivatives should be [] or same size as parameter array");

  if (returnmaps) maps = array(double, nx, ny, d(4)+1);
  //  tot=cube(1,1,)*0.;
  for (i = 1; i<=nx; ++i) {
    for (j = 1; j<=ny; ++j) {
      if (anyof(weight(i,j,))) {
        spectrum = cube(i,j,);
        model = fcn(fcn_x, xs(i,j,), grad, deriv=1);
        if (!is_void(pscale)) grad *= pscale(-,);
        if (!is_void(derivatives)) grad *= derivatives(i, j, - , );
        atom = (model - spectrum) * weight(i,j,);
        //        tot+=atom;
        if (returnmaps) {
          maps(i, j, 0) = sum(atom^2);
        } else {
          res += sum(atom^2);
          gx(i,j,) += (grad * atom(,-))(sum,) *2.;
        }
      }
    }
  }
  //  window,34;
  //  plg, tot;

  if (!is_func(regularisation)) goto out;
  
  xbig=array(double, d(2)*2, d(3)*2, d(4));
  xbig(:d(2), :d(3),) = x;
  xbig(:d(2), d(3)+1:,) = x(,0:1:-1,);
  xbig(d(2)+1:,,) = xbig(d(2):1:-1,,);
  for (k = 1; k<= d(4); ++k) {
    tmp = regularisation(xbig(,,k), g, scale=scale(k), delta=delta(k),
                         returnmap=returnmaps );
    if (returnmaps) {
      maps(,,k) = 4. * tmp(:d(2), :d(3));
    } else {
      res += tmp;
      gx(,,k) +=
        g(1:d(2)  :+1, 1:d(3)  :+1) +
        g(1:d(2)  :+1, 0:d(3)+1:-1) +
        g(0:d(2)+1:-1, 1:d(3)  :+1) +
        g(0:d(2)+1:-1, 0:d(3)+1:-1) ;
    }
  }

  if (returnmaps) return maps;
  
  if (!is_void(decorrelate)) {
    dd= dimsof(decorrelate);
    if (dd(1) != 2) {
      decorrelate=reform(decorrelate, [2, numberof(decorrelate), 1]);
      dd= dimsof(decorrelate);
    }
    npairs = dd(3);
    for (pair = 1; pair <= npairs; ++pair) {
      i1 = long(decorrelate(1, pair));
      i2 = long(decorrelate(2, pair));
      w  = decorrelate(2, pair);
      if (dd(2)>=4) pow=decorrelate(4, pair); else pow = 2;
      xy = x(,,[i1, i2]);
      correl = cubefit.corr(xy, grad, deriv=1);
      res += w * correl^pow;
      gx(,,[i1,i2]) += w*pow*correl^(pow-1)*grad;
    }
  }

 out:
  return res;
}

func cubefit_printer(output, iter, eval, cpu, fx, gnorm, steplen, x)
/* DOCUMENT cubeview.printer method
            fitobj, printer, output, iter, eval, cpu, fx, gnorm, steplen, x
   
 */
{
  local scale, delta, regularisation, decorrelate;
  restore, use, scale, delta, regularisation, decorrelate;
  d = dimsof(x);

  npairs=0;
  if (!is_void(decorrelate)) {
    dd= dimsof(decorrelate);
    if (dd(1) != 2) {
      decorrelate=reform(decorrelate, [2, numberof(decorrelate), 1]);
      dd= dimsof(decorrelate);
    }
    npairs = dd(3);
  }
  
  if (iter==0) {
    write, output, format="# %s",
      "ITER  EVAL   CPU (ms)        FUNC               GNORM   STEPLEN";
    for (k=1; k<=d(4); ++k) write, output, format=" REGUL[%i]", k;
    for (pair = 1; pair <= npairs; ++pair) {
      i1 = long(decorrelate(1, pair));
      i2 = long(decorrelate(2, pair));
      write, format=" CORR[%i,%i]", i1, i2;
    }
    
    write, output, format="\n# %s",
      "---------------------------------------------------------------";
    for (k=1; k<=d(4); ++k) write, output, format="%s","---------";
    for (pair=1; pair<=npairs; ++pair) write, output, format="%s","----------";
    write, output, format="%s\n", "";
  }
  format = " %5d %5d %10.3f  %+-24.15e%-9.1e%-9.1e";
  write, output, format=" %5d %5d %10.3f  %+-24.15e%-9.1e%-9.1e",
    iter, eval, cpu, fx, gnorm, step;

  for (k=1; k<=d(4); ++k)
    write, output, format="%-9.1e", regularisation(x(,,k), g,
                                           scale=scale(k), delta=delta(k) );

  for (pair = 1; pair <= npairs; ++pair) {
    i1 = long(decorrelate(1, pair));
    i2 = long(decorrelate(2, pair));
    w  = decorrelate(2, pair);
    if (dd(2)>=4) pow=decorrelate(4, pair); else pow = 2;
    xy = x(,,[i1, i2]);
    correl = cubefit.corr(xy);
    write, output, format="%-10.1e", w * correl^pow;
  }
  
  write, output, format="%s\n", "";
  
}

func cubefit_fit(x, &fout, &gout,
         xmin=, xmax=, method=, mem=, maxiter=, maxeval=,
         frtol=, fatol=, verb=, quiet=, output=,
         sftol=, sgtol=, sxtol=)
/* DOCUMENT cubefit.fit method
            result = fitobj(fit, x)
         or result = fitobj(fit, x, fout, gout)

   Wrapper around
     result = op_mnb(fitobj.op_f, x, fout, gout, extra=fitobj,
         xmin=xmin, xmax=xmax, method=method, mem=mem, verb=verb, quiet=quiet,
         viewer=fitobj.op_viewer, printer=fitobj.op_printer,
         maxiter=maxiter, maxeval=maxeval,output=output,
         frtol=frtol, fatol=fatol,
         sftol=sftol, sgtol=sgtol, sxtol=sxtol)

   X is rescaled according to fitobj.pscale and fitobj.poffset prior
   to calling op_mnb and scaled back to physical values in
   fitobj.eval. RESULT is also scaled back before being returned.

   KEYWORDS
     This method passes a bunch of keywords through to op_mnb:
         xmin=, xmax=, method=, mem=, maxiter=, maxeval=,
         frtol=, fatol=, verb=, quiet=, output=,
         sftol=, sgtol=, sxtol=

   SEE ALSO
     cubefit, op_mnb
                  
 */
{
  local pscale, poffset, scale, delta, op_f, op_viewer, op_printer;
  restore, use, pscale, poffset, scale, delta, op_f, op_viewer, op_printer;

  nx = (d=dimsof(x))(0);
  
  if (verb &&
      !is_void(op_viewer) && !is_void(view) &&
      op_viewer != noop && view != noop) {
    for (k=0; k<nx; ++k) {
      winkill, k;
      window, k;
      cv_vpaspect,d(2),d(3);
    }
  }

  if (!is_void(poffset)) {
    if (numberof(poffset) != nx) poffset=array(poffset, nx);
    for (k = 1; k <= nx; ++k) x(..,k) -= poffset(k);
    if (!is_void(xmin)) { for (k = 1; k <= nx; ++k) xmin(..,k) -= poffset(k); }
    if (!is_void(xmax)) { for (k = 1; k <= nx; ++k) xmax(..,k) -= poffset(k); }
  }
  
  if (!is_void(pscale)) {
    if (numberof(pscale) != nx) pscale=array(pscale, nx);
    for (k = 1; k <= nx; ++k) x(..,k) /= pscale(k);
    if (!is_void(xmin)) { for (k = 1; k <= nx; ++k) xmin(..,k) /= pscale(k); }
    if (!is_void(xmax)) { for (k = 1; k <= nx; ++k) xmax(..,k) /= pscale(k); }
  }
  
  if (is_void(scale))  scale  = array(1., d(4));
  else if (numberof(scale)==1) scale = array(scale, d(4));

  if (is_void(delta))  delta  = array(1., d(4));
  else if (numberof(delta)==1) delta = array(delta, d(4));
  
  save, use, pscale, poffset, scale, delta;
  result = op_mnb(op_f, x, fout, gout, extra=use(),
                  xmin=xmin, xmax=xmax, method=method, mem=mem, verb=verb, quiet=quiet,
                  viewer=op_viewer, printer=op_printer,
                  maxiter=maxiter, maxeval=maxeval,output=output,
                  frtol=frtol, fatol=fatol,
                  sftol=sftol, sgtol=sgtol, sxtol=sxtol
                  );
  restore, use, pscale, poffset;
  
  if (!is_void(pscale))
    for (k = 1; k <= nx; ++k) result(..,k) *= pscale(k);

  if (!is_void(poffset))
    for (k = 1; k <= nx; ++k) result(..,k) += poffset(k);

  return result;
  
}

func cubefit_op_printer(output, iter, eval, cpu, fx, gnorm, steplen, x, extra) {
  extra, printer, output, iter, eval, cpu, fx, gnorm, steplen, x;
}

func cubefit_op_f(x, &gx, extra)
{
  return extra(eval, x, gx);
}

func cubefit_gmin_f(x, a)
{
  return x(eval, a);
}

func cubefit_gmin_viewer(x, a)
{
  x, view, a;
}

func cubefit_op_viewer(x, extra)
/* DOCUMENT cubefit.op_viewer, x, obj

     A "static function" wrapper around the cubefit.view method, which
     can be used as the VIEWER parameter of the OP_MNB routine from
     optimpack. Equivalent to:

        obj, view, x

   SEE ALSO:
     cubefit, op_mnb, cubefit_view
 */
{
  extra,view, x;
}

func cubefit_markov(object, &grad_obj, scale=, delta=, returnmap=)
/* DOCUMENT cubefit.markov(object,grad_object[,scale=,delta=])
 *
 * delta^2 . sum [|dx|/delta -ln(1+|dx|/delta) + |dy|/delta -ln(1+|dy|/delta)]
 * where |dx| (m,n) = (o(m,n)-o(m-1,n))/scale is the x component of the
 * object gradient normalized by scale.
 *
 * KEYWORDS :
 * scale : Scale factor applied to Do (default value is 1)
 * delta : Threshold on the gradient for switching between linear and
 * quadratic behavour. When delta tends to infinity, the criterion
 * becomes purely quadratic.
 *
 * AUTHOR: Damien Gratadour, borrowed from Yoda.
 */
{

  if (scale==[]) scale = 1.;
  if (delta==[]) delta = 1.;

  dx = (object-roll(object,[1, 0])) / (delta * scale);
  dy = (object-roll(object,[0, 1])) / (delta * scale);

  map = abs(dx)-log(1.+abs(dx))+ abs(dy) - log(1.+abs(dy));

  if (returnmap) return (delta^2)*map;

  crit = (delta^2)*sum(map);

  dx /= (1. + abs(dx));
  dy /= (1. + abs(dy));

  grad_obj = (delta / scale)* (dx-roll(dx,[-1,0])+dy-roll(dy,[0,-1]));

  return crit;
}

func cubefit_l1l2(object, &grad_obj, scale=, delta=, returnmap=)
/* DOCUMENT cubefit.l1l2(object,grad_object[,scale=,delta=])
 *
 * delta^2 . sum [|dx|/delta -ln(1+|dx|/delta) + |dy|/delta -ln(1+|dy|/delta)]
 * where |dx| (m,n) = [o(m,n)-o(m-1,n)]/scale is the x component of the
 * object gradient normalized by scale.
 *
 * KEYWORDS :
 * scale : Scale factor applied to Do (default value is 1)
 * delta : Threshold on the gradient for switching between linear and
 * quadratic behavour. When delta tends to infinity, the criterion
 * becomes purely quadratic.
 *
 * AUTHOR: Damien Gratadour, borrowed from Yoda.
 */
{

  if (!is_set(scale)) scale = 1.;
  if (!is_set(delta)) delta = 1.;

  dx = (object-roll(object,[1,0])) / (delta * scale);
  dy = (object-roll(object,[0,1])) / (delta * scale);
  
  r =sqrt(dx^2+dy^2);

  map = r-log(1.+r);
  
  if (returnmap) return (delta^2) * map;
  
  crit = (delta^2)*sum(map);

  dx /= (1. + r);
  dy /= (1. + r);

  grad_obj = (delta / scale) * (dx-roll(dx,[-1,0])+dy-roll(dy,[0,-1]));

  return crit;
}

func cubefit_corr(xy, &grad, deriv=)
/* DOCUMENT correlation = cubefit.corr(xy [, grad, deriv=1])
    Return the cross-correlation of XY(..,1) and XY(..,2)
 */
{
  x = xy(..,1); y = xy(..,2);
  d = dimsof(x);
  n = numberof(x);
  u = n*sum(x*y) - (sx=sum(x))*(sy=sum(y));
  a = n*sum(x^2) - sx^2;
  b = n*sum(y^2) - sy^2;
  v = sqrt(a*b);
  res = v ? u/v : 1.;
  
  if (deriv) {
    if (v) {
      gx = (n*y -sy - u*(n*x -sx)/a) / v;
      gy = (n*x -sx - u*(n*y -sy)/b) / v;
      grad=[gx, gy];
    }
    else grad = array(1., d, 2);
  }
  
  return res;
}

cubefit = save(
               //methods
               new=cubefit_new, view=cubefit_view, model=cubefit_model,
               printer=cubefit_printer, fit=cubefit_fit, eval=cubefit_eval,
               // static functions
               l1l2=cubefit_l1l2, markov=cubefit_markov, corr=cubefit_corr,
               op_f=cubefit_op_f, op_viewer=cubefit_op_viewer,
               op_printer=cubefit_op_printer
               );

// clean namespace
restore, scratch;


/*
// fog' = g'.f'og
// (1/v)' = v'.(1/x)'ov =-v'/v^2
// (u.v)' = u'.v + u.v'
// (u/v)' = u'/v - u.v'/v^2 = (u'.v - u.v') / v^2.

u(xj) = Sum [(xi-<x>)(yi-<y>)] = Sum [xi.(yi-<y>)] - <x> Sum (yi-<y>)
u'(xj) = (yj-<y>) - d<x>/dxj * Sum(yi-<y>) = (yj-<y>) - < (yi -<y>) >

v(xj) = sqrt( Sum (xi-<x>)^2 ) * sqrt(Sum (yi-<y>)^2)


*/
