document.addEventListener('DOMContentLoaded', function () {
  const form = document.getElementById('vpnForm');
  if (form) {
    form.addEventListener('submit', handleFormSubmit);
  }

  // Real-time validation
  document.getElementById('speed')?.addEventListener('blur', () =>
    validateField('speed', 1, 1000, 'Please enter a speed between 1 and 1000 Mbps')
  );
  document.getElementById('price')?.addEventListener('blur', () =>
    validateField('price', 0, 100, 'Please enter a price between $0 and $100')
  );
  document.getElementById('max_devices')?.addEventListener('blur', () =>
    validateField('max_devices', 1, 100, 'Please enter a number between 1 and 100')
  );
});

async function handleFormSubmit(event) {
  event.preventDefault();
  const form = event.target;
  document.querySelectorAll('.error').forEach(el => el.textContent = '');
  let isValid = true;

  // Validate number inputs
  isValid = validateField('speed', 1, 1000, 'Please enter a speed between 1 and 1000 Mbps') && isValid;
  isValid = validateField('price', 0, 100, 'Please enter a price between $0 and $100') && isValid;
  isValid = validateField('max_devices', 1, 100, 'Please enter a number between 1 and 100') && isValid;

  // Validate selects
  isValid = validateSelect('trial_available', 'This field is required') && isValid;
  isValid = validateSelect('logging_policy', 'This field is required') && isValid;
  isValid = validateSelect('encryption', 'This field is required') && isValid;
  isValid = validateSelect('country', 'This field is required') && isValid;

  if (!isValid) {
    const firstError = document.querySelector('.error:not(:empty)');
    if (firstError) {
      firstError.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    return;
  }

  // Show loading state
  const submitBtn = form.querySelector('button[type="submit"]');
  submitBtn.disabled = true;
  submitBtn.innerHTML = '<span class="spinner"></span> Processing...';

  try {
    const response = await fetch(form.action, {
      method: 'POST',
      body: new FormData(form),
      headers: { Accept: 'text/html' }
    });

    if (!response.ok) throw new Error('Network response was not ok');
    const html = await response.text();
    document.body.innerHTML = html;
  } catch (error) {
    console.error('Error:', error);
    showFormError('An error occurred while processing your request. Please try again.');
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Get VPN Recommendations';
  }
}

function validateField(id, min, max, message) {
  const input = document.getElementById(id);
  const errorEl = document.getElementById(`${id}Error`);
  if (!input || !errorEl) return true;

  const raw = input.value.trim();
  if (raw === '') {
    errorEl.textContent = 'This field is required';
    return false;
  }

  const value = parseFloat(raw);
  if (isNaN(value) || value < min || value > max) {
    errorEl.textContent = message;
    return false;
  }

  errorEl.textContent = '';
  return true;
}

function validateSelect(id, message) {
  const select = document.getElementById(id);
  const errorEl = document.getElementById(`${id}Error`);
  if (!select || !errorEl) return true;

  if (!select.value || select.value.trim() === '') {
    errorEl.textContent = message;
    return false;
  }

  errorEl.textContent = '';
  return true;
}

function showFormError(message) {
  const form = document.getElementById('vpnForm');
  if (!form) return;

  let errorContainer = document.getElementById('formError');
  if (!errorContainer) {
    errorContainer = document.createElement('div');
    errorContainer.id = 'formError';
    errorContainer.className = 'error';
    form.prepend(errorContainer);
  }

  errorContainer.textContent = message;
}
