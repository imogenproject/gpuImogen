function f = evaluatePadeApproximant(cv, m, n, x)
% Given the power series coefficients c_n defining a power series, computes the P_a(x)/Q_b(x)
% Pade approximant and evaluates it at the given x. Requires a+b<4.

M = numel(cv);

a = cv(1);
if M >= 2; b = cv(2); end
if M >= 3; c = cv(3); end
if M >= 4; d = cv(4); end
if M >= 5; e = cv(5); end
if M >= 6; f = cv(5); end
if M >= 7; g = cv(6); end

if m+n > M; error('Pade approximant order m+n cannot exceed power series order.'); end

% Prevent attempt to use inverse polynomial with zero-valued argument
if(abs(a) < 1e-8); m = m + n; n = 0; end

P = 10*n+m;

switch P;
  case 0;  f = a;
  case 1;  f = a + b*x;
  case 2;  f = a + x.*(b + c*x);
  case 3;  f = a + x.*(b + x.*(c + d*x));
  case 4;  f = a + x.*(b + x.*(c + x.*(d + e*x)));
  case 5;  f = a + x.*(b + x.*(c + x.*(d + x.*(e + f*x))));
  case 6;  f = a + x.*(b + x.*(c + x.*(d + x.*(e + x.*(f + g*x)))));

  case 10; f = a^2/(a - b*x);
  case 11; f = (a*b + b^2.*x - a*c*x)/(b - c*x);
  case 12; f = (a*(c - d*x) + x.*(c^2.*x + b*(c - d*x)))/(c - d*x);
  case 13; f = (a*(d - e*x) + x.*(b*(d - e*x) + x.*(c*d + d^2.*x - c*e*x)))/(d - e*x);
  case 14; f = (a*(e - f*x) + x.*(b*(e - f*x) + x.*(c*(e - f*x) + x.*(d*e + e^2.*x - d*f*x))))/(e - f*x);
  case 15; f = (a*(f - g*x) + x.*(b*(f - g*x) + x.*(c*(f - g*x) + x.*(d*(f - g*x) + x.*(e*f + f^2.*x - e*g*x)))))/(f - g*x);

  case 20; f = a^3/(a^2 + b^2.*x.^2 - a*x.*(b + c*x));
  case 21; f = (b^3.*x + a*b*(b - 2.*c*x) + a^2.*(-c + d*x))/(b^2 - a*c + a*d*x + c^2.*x.^2 - b*x.*(c + d*x));
  case 22; f = (x.*(c^3.*x + b*c*(c - 2.*d*x) + b^2.*(-d + e*x)) + a*(c^2 - b*d + b*e*x + d^2.*x.^2 - c*x.*(d + e*x)))/(c^2 - b*d + b*e*x + d^2.*x.^2 - c*x.*(d + e*x));
  case 23; f = (a*(d^2 - c*e + c*f*x + e^2.*x.^2 - d*x.*(e + f*x)) + x.*(x.*(d^3.*x + c*d*(d - 2.*e*x) + c^2.*(-e + f*x)) + b*(d^2 - c*e + c*f*x + e^2.*x.^2 - d*x.*(e + f*x))))/(d^2 - c*e + c*f*x + e^2.*x.^2 - d*x.*(e + f*x));
  case 24; f = (a*(e^2 - d*f + d*g*x + f^2.*x.^2 - e*x.*(f + g*x)) + x.*(b*(e^2 - d*f + d*g*x + f^2.*x.^2 - e*x.*(f + g*x)) + x.*(x.*(e^3.*x + d*e*(e - 2.*f*x) + d^2.*(-f + g*x)) + c*(e^2 - d*f + d*g*x + f^2.*x.^2 - e*x.*(f + g*x)))))/(e^2 - d*f + d*g*x + f^2.*x.^2 - e*x.*(f + g*x));

  case 30; f = a^4/(a^3 - b^3.*x.^3 + a*b*x.^2.*(b + 2.*c*x) - a^2.*x.*(b + x.*(c + d*x)));
  case 31; f = (b^4.*x + a*b^2.*(b - 3.*c*x) + a^2.*(-2.*b*c + c^2.*x + 2.*b*d*x) + a^3.*(d - e*x))/(b^3 + a^2.*d + a*(c^2 - a*e).*x - a*c*d*x.^2 - (c^3 + a*d^2 - a*c*e).*x.^3 - b^2.*x.*(c + x.*(d + e*x)) + b*(c*x.^2.*(c + 2.*d*x) + a*(-2.*c + x.*(d + e*x))));
  case 32; f = -((a*(-c^3 + 2.*b*c*d - a*d^2 - b^2.*e + a*c*e) + (-(b*(c^3 + 2.*a*d^2)) - b^3.*e + b^2.*(2.*c*d + a*f) + a*(c^2.*d + a*d*e - a*c*f)).*x - (c^4 + (b*d - a*e).^2 - c^2.*(3.*b*d + 2.*a*e) - (b^3 + a^2.*d).*f + 2.*c*(b^2.*e + a*(d^2 + b*f))).*x.^2)/(c^3 + b^2.*e + b*(d^2 - b*f).*x - b*d*e*x.^2 - (d^3 + b*e^2 - b*d*f).*x.^3 - c^2.*x.*(d + x.*(e + f*x)) + a*(d^2 + e^2.*x.^2 - d*x.*(e + f*x)) + c*(-(a*e) + x.*(a*f + d*x.*(d + 2.*e*x)) + b*(-2.*d + x.*(e + f*x)))));
  case 33; f = (x.*(b*(d^3 + b*e^2 + c^2.*f - d*(2.*c*e + b*f)) + (c*(d^3 + 2.*b*e^2) + c^3.*f - c^2.*(2.*d*e + b*g) + b*(-(e*(d^2 + b*f)) + b*d*g)).*x + (d^4 + (c*e - b*f).^2 - d^2.*(3.*c*e + 2.*b*f) - (c^3 + b^2.*e).*g + 2.*d*(c^2.*f + b*(e^2 + c*g))).*x.^2) + a*(d^3 + c^2.*f + c*(e^2 - c*g).*x - c*e*f*x.^2 - (e^3 + c*f^2 - c*e*g).*x.^3 - d^2.*x.*(e + x.*(f + g*x)) + b*(e^2 + f^2.*x.^2 - e*x.*(f + g*x)) + d*(-(b*f) + x.*(b*g + e*x.*(e + 2.*f*x)) + c*(-2.*e + x.*(f + g*x)))))/ (d^3 + c^2.*f + c*(e^2 - c*g).*x - c*e*f*x.^2 - (e^3 + c*f^2 - c*e*g).*x.^3 - d^2.*x.*(e + x.*(f + g*x)) + b*(e^2 + f^2.*x.^2 - e*x.*(f + g*x)) + d*(-(b*f) + x.*(b*g + e*x.*(e + 2.*f*x)) + c*(-2.*e + x.*(f + g*x))));

  case 40; f = a^5/(a^4 + b^4.*x.^4 - a*b^2.*x.^3.*(b + 3.*c*x) + a^2.*x.^2.*(b^2 + c^2.*x.^2 + 2.*b*x.*(c + d*x)) - a^3.*x.*(b + x.*(c + x.*(d + e*x))));
  case 41; f = (b^5.*x + a*b^3.*(b - 4.*c*x) + 3.*a^2.*b*(-(b*c) + c^2.*x + b*d*x) + a^3.*(c^2 + 2.*b*d - 2.*(c*d + b*e).*x) + a^4.*(-e + f*x))/ (b^4 + c^4.*x.^4 + a^3.*(-e + f*x) + a*c*x.^2.*(-c^2 + 2.*d^2.*x.^2 + c*x.*(d - 2.*e*x)) + a^2.*(c^2 + x.^3.*(d*e + e^2.*x - d*f*x) + c*x.*(-2.*d + x.*(e - f*x))) - b^3.*x.*(c + x.*(d + x.*(e + f*x))) + b*(-(c^2.*x.^3.*(c + 3.*d*x)) + a^2.*(2.*d - x.*(e + f*x)) + 2.*a*x.*(-(d*x.^2.*(d + e*x)) + c*(c + f*x.^3))) + b^2.*(x.^2.*(c^2 + d^2.*x.^2 + 2.*c*x.*(d + e*x)) + a*(-3.*c + x.*(d + x.*(e + f*x)))));
  case 42; f = (x.*(c^5.*x + b*c^3.*(c - 4.*d*x) + 3.*b^2.*c*(-(c*d) + d^2.*x + c*e*x) + b^3.*(d^2 + 2.*c*e - 2.*(d*e + c*f).*x) + b^4.*(-f + g*x)) + a^3.*(e^2 - d*f + d*g*x + f^2.*x.^2 - e*x.*(f + g*x)) + a^2.*(d^2.*x.*(-d + e*x) + 2.*c*(d^2 + b*f - b*g*x + (e^2 - 2.*d*f).*x.^2) + c^2.*(-2.*e + x.*(f + g*x)) + 2.*b*(e*x.*(e - f*x) + d*(-e + g*x.^2))) + a*(c^4 - c^3.*x.*(d + 3.*e*x) + b*(-2.*d^3.*x.^2 + b^2.*(-f + g*x) + b*(d^2 - 4.*d*e*x + (e^2 + 2.*d*f).*x.^2)) + c^2.*(3.*d^2.*x.^2 - b*(3.*d + x.*(e - 4.*f*x))) + b*c*(2.*d*x.*(2.*d - e*x) + b*(2.*e + x.*(f - 3.*g*x)))))/ (c^4 + a^2.*(e^2 - d*f) + a*(-d^3 - a*e*f + a*d*g).*x + a*(d^2.*e + a*(f^2 - e*g)).*x.^2 + a*d*(-e^2 + d*f).*x.^3 + (d^4 - a*e^3 + 2.*a*d*e*f - a*d^2.*g).*x.^4 + b^3.*(-f + g*x) + b*(-2.*a*d*e + a*(e^2 + d*f).*x - (d^3 + a*e*f - a*d*g).*x.^2 + (d^2.*e - a*f^2 + a*e*g).*x.^3 + 2.*d*(e^2 - d*f).*x.^4) + b^2.*(d^2 + x.^3.*(f^2.*x + e*(f - g*x)) + d*x.*(-2.*e + x.*(f - g*x))) - c^3.*x.*(d + x.*(e + x.*(f + g*x))) + c^2.*(-2.*a*e + x.*(a*(f + g*x) + x.*(d^2 + e^2.*x.^2 + 2.*d*x.*(e + f*x))) + b*(-3.*d + x.*(e + x.*(f + g*x)))) + c*(-(d^2.*x.^3.*(d + 3.*e*x)) + 2.*b*x.*(d^2 + d*g*x.^3 - e*x.^2.*(e + f*x)) + b^2.*(2.*e - x.*(f + g*x)) + a*(2.*d^2 + 2.*b*(f - g*x) - d*x.^2.*(3.*f + g*x) + x.^2.*(e^2 - f^2.*x.^2 + e*x.*(f + g*x)))));

  case 50; f = a^6/(a^5 - b^5.*x.^5 + a*b^3.*x.^4.*(b + 4.*c*x) - a^2.*b*x.^3.*(b^2 + 3.*c^2.*x.^2 + 3.*b*x.*(c + d*x)) + a^3.*x.^2.*(b^2 + c*x.^2.*(c + 2.*d*x) + 2.*b*x.*(c + x.*(d + e*x))) - a^4.*x.*(b + x.*(c + x.*(d + x.*(e + f*x)))));
  case 51; f = (b^6.*x + a*b^4.*(b - 5.*c*x) + 2.*a^2.*b^2.*(-2.*b*c + 3.*c^2.*x + 2.*b*d*x) + a^3.*(3.*b*(c^2 + b*d) - (c^3 + 6.*b*c*d + 3.*b^2.*e).*x) + a^4.*(-2.*(c*d + b*e) + (d^2 + 2.*c*e + 2.*b*f).*x) + a^5.*(f - g*x))/ (b^5 - c^5.*x.^5 + a^4.*(f - g*x) + a*c^2.*x.^3.*(c^2 - 3.*d^2.*x.^2 + c*x.*(-d + 3.*e*x)) + a^3.*(x.*(d^2 + d*x.^2.*(-f + g*x) + x.^3.*(-(e*f) - f^2.*x + e*g*x)) + c*(-2.*d + x.*(2.*e - f*x + g*x.^2))) - a^2.*x.*(c^3 - c*d^2.*x.^2 + 2.*c*(e^2 - 2.*d*f).*x.^4 + d^2.*x.^3.*(d + e*x) + c^2.*x.*(-2.*d + x.*(2.*e - f*x + g*x.^2))) - b^4.*x.*(c + x.*(d + x.*(e + x.*(f + g*x)))) + b*(c^3.*x.^4.*(c + 4.*d*x) + a^3.*(-2.*e + x.*(f + g*x)) + a*x.^2.*(-2.*c^3 + 2.*d^3.*x.^3 + 2.*c*d*x.^2.*(2.*d + e*x) - c^2.*x.*(d + x.*(e + 4.*f*x))) + a^2.*(3.*c^2 - 2.*c*(2.*d*x + g*x.^4) + x.^2.*(-d^2 + 2.*e*x.^2.*(e + f*x) + 2.*d*x.*(e - g*x.^2)))) + b^3.*(x.^2.*(c^2 + d*x.^2.*(d + 2.*e*x) + 2.*c*x.*(d + x.*(e + f*x))) + a*(-4.*c + x.*(d + x.*(e + x.*(f + g*x))))) + b^2.*(-(c*x.^3.*(c^2 + 3.*d^2.*x.^2 + 3.*c*x.*(d + e*x))) + a^2.*(3.*d - x.*(e + x.*(f + g*x))) + a*x.*(3.*c^2 - x.^2.*(2.*d^2 + 4.*d*e*x + (e^2 + 2.*d*f).*x.^2) + c*x.*(d + x.*(e + x.*(f + 3.*g*x))))));

  case 60; f = a^7/(a^6 + b^6.*x.^6 - a*b^4.*x.^5.*(b + 5.*c*x) + a^2.*b^2.*x.^4.*(b^2 + 6.*c^2.*x.^2 + 4.*b*x.*(c + d*x)) - a^3.*x.^3.*(b^3 + c^3.*x.^3 + 3.*b*c*x.^2.*(c + 2.*d*x) + 3.*b^2.*x.*(c + x.*(d + e*x))) + a^4.*x.^2.*(b^2 + x.^2.*(c^2 + d^2.*x.^2 + 2.*c*x.*(d + e*x)) + 2.*b*x.*(c + x.*(d + x.*(e + f*x)))) - a^5.*x.*(b + x.*(c + x.*(d + x.*(e + x.*(f + g*x))))));

  otherwise; error('Requested order falls outside available formulae.');
end

end
