import React, { useState } from 'react';
import { 
  signInWithEmailAndPassword, 
  createUserWithEmailAndPassword,
  sendPasswordResetEmail
} from 'firebase/auth';
import { auth } from './firebase';
import { Mail, Lock, Eye, EyeOff, Loader2, ArrowRight, UserPlus, LogIn, KeyRound } from 'lucide-react';

interface AuthScreenProps {
  onAuthSuccess: () => void;
}

export const AuthScreen: React.FC<AuthScreenProps> = ({ onAuthSuccess }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [isForgotPassword, setIsForgotPassword] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const getErrorMessage = (code: string): string => {
    switch (code) {
      case 'auth/invalid-email':
        return 'Indirizzo email non valido.';
      case 'auth/user-disabled':
        return 'Questo account è stato disabilitato.';
      case 'auth/user-not-found':
        return 'Nessun account trovato con questa email.';
      case 'auth/wrong-password':
        return 'Password non corretta.';
      case 'auth/invalid-credential':
        return 'Credenziali non valide. Controlla email e password.';
      case 'auth/email-already-in-use':
        return 'Questa email è già registrata.';
      case 'auth/weak-password':
        return 'La password deve essere di almeno 6 caratteri.';
      case 'auth/too-many-requests':
        return 'Troppi tentativi. Riprova più tardi.';
      default:
        return 'Si è verificato un errore. Riprova.';
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccessMessage(null);

    if (!email || !password) {
      setError('Inserisci email e password.');
      return;
    }

    if (!isLogin && password !== confirmPassword) {
      setError('Le password non coincidono.');
      return;
    }

    setLoading(true);

    try {
      if (isLogin) {
        await signInWithEmailAndPassword(auth, email, password);
      } else {
        await createUserWithEmailAndPassword(auth, email, password);
      }
      onAuthSuccess();
    } catch (err: any) {
      console.error('Auth error:', err);
      setError(getErrorMessage(err.code));
    } finally {
      setLoading(false);
    }
  };

  const handleForgotPassword = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccessMessage(null);

    if (!email) {
      setError('Inserisci la tua email.');
      return;
    }

    setLoading(true);

    try {
      await sendPasswordResetEmail(auth, email);
      setSuccessMessage('Email di reset inviata! Controlla la tua casella di posta.');
    } catch (err: any) {
      console.error('Reset password error:', err);
      setError(getErrorMessage(err.code));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'linear-gradient(135deg, #fdf4ff 0%, #faf5ff 25%, #f5f3ff 50%, #eff6ff 75%, #f0fdfa 100%)',
      padding: '20px'
    }}>
      <div style={{
        width: '100%',
        maxWidth: '420px',
        backgroundColor: 'rgba(255,255,255,0.85)',
        backdropFilter: 'blur(12px)',
        borderRadius: '24px',
        padding: '40px',
        boxShadow: '0 25px 50px -12px rgba(0,0,0,0.1)',
        border: '1px solid rgba(255,255,255,0.8)'
      }}>
        {/* Logo */}
        <div style={{ textAlign: 'center', marginBottom: '32px' }}>
          <div style={{
            width: '72px',
            height: '72px',
            borderRadius: '50%',
            background: 'linear-gradient(135deg, #9333ea 0%, #7c3aed 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 16px',
            boxShadow: '0 8px 24px rgba(147, 51, 234, 0.3)'
          }}>
            <span style={{ fontSize: '32px' }}>🤗</span>
          </div>
          <h1 style={{
            fontSize: '28px',
            fontWeight: 800,
            color: '#1e293b',
            marginBottom: '8px',
            fontFamily: "'Nunito', sans-serif"
          }}>
            Ti Ascolto
          </h1>
          <p style={{
            fontSize: '14px',
            color: '#64748b',
            fontWeight: 500
          }}>
            {isForgotPassword 
              ? 'Recupera la tua password' 
              : isLogin 
                ? 'Accedi al tuo account' 
                : 'Crea il tuo account'}
          </p>
        </div>

        {/* Error/Success Messages */}
        {error && (
          <div style={{
            padding: '12px 16px',
            backgroundColor: '#fef2f2',
            border: '1px solid #fecaca',
            borderRadius: '12px',
            color: '#dc2626',
            fontSize: '14px',
            marginBottom: '20px',
            textAlign: 'center'
          }}>
            {error}
          </div>
        )}

        {successMessage && (
          <div style={{
            padding: '12px 16px',
            backgroundColor: '#f0fdf4',
            border: '1px solid #bbf7d0',
            borderRadius: '12px',
            color: '#16a34a',
            fontSize: '14px',
            marginBottom: '20px',
            textAlign: 'center'
          }}>
            {successMessage}
          </div>
        )}

        {/* Form */}
        <form onSubmit={isForgotPassword ? handleForgotPassword : handleSubmit}>
          {/* Email */}
          <div style={{ marginBottom: '16px' }}>
            <label style={{
              display: 'block',
              fontSize: '12px',
              fontWeight: 600,
              color: '#64748b',
              marginBottom: '8px',
              textTransform: 'uppercase',
              letterSpacing: '0.05em'
            }}>
              Email
            </label>
            <div style={{ position: 'relative' }}>
              <Mail size={18} style={{
                position: 'absolute',
                left: '16px',
                top: '50%',
                transform: 'translateY(-50%)',
                color: '#94a3b8'
              }} />
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="la.tua@email.com"
                style={{
                  width: '100%',
                  padding: '14px 16px 14px 48px',
                  fontSize: '15px',
                  border: '1px solid #e2e8f0',
                  borderRadius: '12px',
                  outline: 'none',
                  backgroundColor: 'rgba(255,255,255,0.8)',
                  transition: 'border-color 0.2s, box-shadow 0.2s',
                  boxSizing: 'border-box'
                }}
                onFocus={(e) => {
                  e.target.style.borderColor = '#9333ea';
                  e.target.style.boxShadow = '0 0 0 3px rgba(147, 51, 234, 0.1)';
                }}
                onBlur={(e) => {
                  e.target.style.borderColor = '#e2e8f0';
                  e.target.style.boxShadow = 'none';
                }}
              />
            </div>
          </div>

          {/* Password (hidden for forgot password) */}
          {!isForgotPassword && (
            <div style={{ marginBottom: '16px' }}>
              <label style={{
                display: 'block',
                fontSize: '12px',
                fontWeight: 600,
                color: '#64748b',
                marginBottom: '8px',
                textTransform: 'uppercase',
                letterSpacing: '0.05em'
              }}>
                Password
              </label>
              <div style={{ position: 'relative' }}>
                <Lock size={18} style={{
                  position: 'absolute',
                  left: '16px',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  color: '#94a3b8'
                }} />
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  style={{
                    width: '100%',
                    padding: '14px 48px 14px 48px',
                    fontSize: '15px',
                    border: '1px solid #e2e8f0',
                    borderRadius: '12px',
                    outline: 'none',
                    backgroundColor: 'rgba(255,255,255,0.8)',
                    transition: 'border-color 0.2s, box-shadow 0.2s',
                    boxSizing: 'border-box'
                  }}
                  onFocus={(e) => {
                    e.target.style.borderColor = '#9333ea';
                    e.target.style.boxShadow = '0 0 0 3px rgba(147, 51, 234, 0.1)';
                  }}
                  onBlur={(e) => {
                    e.target.style.borderColor = '#e2e8f0';
                    e.target.style.boxShadow = 'none';
                  }}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  style={{
                    position: 'absolute',
                    right: '16px',
                    top: '50%',
                    transform: 'translateY(-50%)',
                    background: 'none',
                    border: 'none',
                    cursor: 'pointer',
                    color: '#94a3b8',
                    padding: '4px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}
                >
                  {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                </button>
              </div>
            </div>
          )}

          {/* Confirm Password (only for registration) */}
          {!isLogin && !isForgotPassword && (
            <div style={{ marginBottom: '20px' }}>
              <label style={{
                display: 'block',
                fontSize: '12px',
                fontWeight: 600,
                color: '#64748b',
                marginBottom: '8px',
                textTransform: 'uppercase',
                letterSpacing: '0.05em'
              }}>
                Conferma Password
              </label>
              <div style={{ position: 'relative' }}>
                <Lock size={18} style={{
                  position: 'absolute',
                  left: '16px',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  color: '#94a3b8'
                }} />
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="••••••••"
                  style={{
                    width: '100%',
                    padding: '14px 16px 14px 48px',
                    fontSize: '15px',
                    border: '1px solid #e2e8f0',
                    borderRadius: '12px',
                    outline: 'none',
                    backgroundColor: 'rgba(255,255,255,0.8)',
                    transition: 'border-color 0.2s, box-shadow 0.2s',
                    boxSizing: 'border-box'
                  }}
                  onFocus={(e) => {
                    e.target.style.borderColor = '#9333ea';
                    e.target.style.boxShadow = '0 0 0 3px rgba(147, 51, 234, 0.1)';
                  }}
                  onBlur={(e) => {
                    e.target.style.borderColor = '#e2e8f0';
                    e.target.style.boxShadow = 'none';
                  }}
                />
              </div>
            </div>
          )}

          {/* Forgot Password Link (only for login) */}
          {isLogin && !isForgotPassword && (
            <div style={{ textAlign: 'right', marginBottom: '20px' }}>
              <button
                type="button"
                onClick={() => {
                  setIsForgotPassword(true);
                  setError(null);
                  setSuccessMessage(null);
                }}
                style={{
                  background: 'none',
                  border: 'none',
                  color: '#7c3aed',
                  fontSize: '13px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  textDecoration: 'none'
                }}
              >
                Password dimenticata?
              </button>
            </div>
          )}

          {/* Submit Button */}
          <button
            type="submit"
            disabled={loading}
            style={{
              width: '100%',
              padding: '16px 24px',
              fontSize: '15px',
              fontWeight: 700,
              color: 'white',
              background: loading 
                ? '#a1a1aa' 
                : 'linear-gradient(135deg, #9333ea 0%, #7c3aed 100%)',
              border: 'none',
              borderRadius: '12px',
              cursor: loading ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '10px',
              boxShadow: loading ? 'none' : '0 8px 24px rgba(147, 51, 234, 0.3)',
              transition: 'all 0.2s'
            }}
          >
            {loading ? (
              <>
                <Loader2 size={20} className="animate-spin" style={{ animation: 'spin 1s linear infinite' }} />
                Attendere...
              </>
            ) : isForgotPassword ? (
              <>
                <KeyRound size={20} />
                Invia email di reset
              </>
            ) : isLogin ? (
              <>
                <LogIn size={20} />
                Accedi
              </>
            ) : (
              <>
                <UserPlus size={20} />
                Registrati
              </>
            )}
          </button>
        </form>

        {/* Toggle Login/Register */}
        <div style={{
          marginTop: '24px',
          textAlign: 'center',
          paddingTop: '24px',
          borderTop: '1px solid #e2e8f0'
        }}>
          {isForgotPassword ? (
            <button
              onClick={() => {
                setIsForgotPassword(false);
                setError(null);
                setSuccessMessage(null);
              }}
              style={{
                background: 'none',
                border: 'none',
                color: '#64748b',
                fontSize: '14px',
                cursor: 'pointer'
              }}
            >
              <ArrowRight size={16} style={{ 
                transform: 'rotate(180deg)', 
                verticalAlign: 'middle', 
                marginRight: '6px' 
              }} />
              Torna al login
            </button>
          ) : (
            <p style={{ color: '#64748b', fontSize: '14px', margin: 0 }}>
              {isLogin ? "Non hai un account? " : "Hai già un account? "}
              <button
                onClick={() => {
                  setIsLogin(!isLogin);
                  setError(null);
                  setPassword('');
                  setConfirmPassword('');
                }}
                style={{
                  background: 'none',
                  border: 'none',
                  color: '#7c3aed',
                  fontWeight: 700,
                  cursor: 'pointer',
                  fontSize: '14px'
                }}
              >
                {isLogin ? 'Registrati' : 'Accedi'}
              </button>
            </p>
          )}
        </div>

        {/* Footer */}
        <div style={{
          marginTop: '24px',
          textAlign: 'center',
          fontSize: '11px',
          color: '#94a3b8'
        }}>
          © Effetre Properties IA Division 2025
        </div>
      </div>

      {/* CSS for spin animation */}
      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default AuthScreen;
